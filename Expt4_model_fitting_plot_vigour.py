# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, LIVEAMP_DEVICE_NAME, UNITY_DEVICE_NAME, EEGLAB_NAME
from core.plot import add_significance_bar_hue, get_x_positions, significance_converter
from core.experiment_data import set_expt

EXPT_NAME = 'Expt4'

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series
from core.utils import save_cache, load_cache
from core.individual_subject import get_series_from_control
import json

start_trial_for_analysis = 6
end_trial_for_analysis = 24
pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]

set_expt('Expt4')
exp_data = make_experiment_data(exclude_participants=['SUB14', 'SUB20'], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME])
subjects = list(exp_data.keys())

with open('model_fitting_results/single_vigour/FalseTrue_fitted.json') as f:
    tonic_data_dict = json.load(f)

with open('model_fitting_results/single_vigour/TrueFalse_fitted.json') as f:
    no_tonic_data_dict = json.load(f)

tonic_data_list = []
no_tonic_data_list = []
for k in subjects:
    tonic_data_list.append(tonic_data_dict[k])
    no_tonic_data_list.append(no_tonic_data_dict[k])
tonic_data_array = np.array(tonic_data_list)
no_tonic_data_array = np.array(no_tonic_data_list)

def myboxplot(ax, amp1, amp2, pal, label1, label2, p_val):
    ort = 'v'
    print(np.array(amp1 + amp2).shape)
    analyse_pd = pd.DataFrame({'Data': np.array(amp1 + amp2), 'Group': [label1] * len(amp1) + [label2] * len(amp2)})
    dx = 'Group'
    dy = 'Data'
    ax=pt.half_violinplot( x = dx, y = dy, data = analyse_pd, palette=pal,
        bw = 0.5, cut = 0.,scale = "area", width = 0.5, inner = None, 
        orient = ort, alpha = 0.8, linewidth=0, ax=ax)

    ax=sns.stripplot( x = dx, y = dy, data = analyse_pd, palette=pal,
        edgecolor = "white",size = 6, jitter = 1, zorder = 0, 
        orient = ort, ax=ax)

    ax=sns.boxplot( x = dx, y = dy, data = analyse_pd, palette=pal,
        width = .25, zorder = 10, showcaps = True, showfliers=True,
        whiskerprops = {'linewidth':2, "zorder":10}, 
        saturation = 1, orient = ort, ax=ax)
    
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))

    max_value = analyse_pd['Data'].to_numpy()
    significance_bar_vertical_len = 1
    max_value = np.nanmax(max_value) + significance_bar_vertical_len * 3
    add_significance_bar_hue(ax, 0, 1, max_value, significance_bar_vertical_len, significance_converter(p_val), p_val)

    ax.tick_params(axis='both', labelsize=18)
    sns.set_theme()
    sns.set_style("ticks")
    sns.despine(offset=0, trim=True, ax=ax)

figure = plt.figure(figsize=(21, 8))

axs = figure.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]}, sharey=True)

ax = axs[0]

ax.text(-0.1, 1.1, 'a', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')

# 3rd parameter denotes vigour constant
vigour_test = stats.ttest_rel(tonic_data_array.T[2], no_tonic_data_array.T[2], alternative='two-sided')
print(np.mean(tonic_data_array.T[2]), np.std(tonic_data_array.T[2]))
print(np.mean(no_tonic_data_array.T[2]), np.std(no_tonic_data_array.T[2]))
print(vigour_test)
myboxplot(ax, no_tonic_data_array.T[2].tolist(), tonic_data_array.T[2].tolist(), 'Set2', 'No tonic pain', 'With tonic pain', vigour_test.pvalue)

ax.set_ylabel('Vigour constant $C_v$', fontsize=24)
ax.set_xlabel('')
ax.tick_params(axis='both', labelsize=24)

import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
from rpy2.robjects.conversion import localconverter

base = importr("base")
lme4 = importr('lme4')
lmer_test = importr('lmerTest')
rstats = importr('stats')
broom = importr('broom.mixed')
generics = importr('generics')
rstatix = importr("rstatix")
import rpy2.robjects.lib.ggplot2 as ggplot2

with open('model_fitting_results/separate_vigour/FalseTrue_fitted.json') as f:
    tonic_data = json.load(f)

with open('model_fitting_results/separate_vigour/TrueFalse_fitted.json') as f:
    no_tonic_data = json.load(f)

test_param = 2
tonic_nopain_list_vigour = []
tonic_lowpain_list_vigour = []
tonic_highpain_list_vigour = []
notonic_nopain_list_vigour = []
notonic_lowpain_list_vigour = []
notonic_highpain_list_vigour = []
vigour_list = []
tonic_list = []
phasic_list = []
subject_list = []
transform = lambda x: x
for subject_id in tonic_data.keys():
    tonic_nopain_list_vigour.append(transform(tonic_data[subject_id][test_param]))
    notonic_nopain_list_vigour.append(transform(no_tonic_data[subject_id][test_param]))
    tonic_lowpain_list_vigour.append(transform(tonic_data[subject_id][test_param + 1]))
    notonic_lowpain_list_vigour.append(transform(no_tonic_data[subject_id][test_param + 1]))
    tonic_highpain_list_vigour.append(transform(tonic_data[subject_id][test_param + 2]))
    notonic_highpain_list_vigour.append(transform(no_tonic_data[subject_id][test_param + 2]))

    vigour_list.append(transform(no_tonic_data[subject_id][test_param]))
    tonic_list.append('No tonic pain')
    phasic_list.append('No pain')
    vigour_list.append(transform(tonic_data[subject_id][test_param]))
    tonic_list.append('With tonic pain')
    phasic_list.append('No pain')
    vigour_list.append(transform(no_tonic_data[subject_id][test_param + 1]))
    tonic_list.append('No tonic pain')
    phasic_list.append('Low pain')
    vigour_list.append(transform(tonic_data[subject_id][test_param + 1]))
    tonic_list.append('With tonic pain')
    phasic_list.append('Low pain')
    vigour_list.append(transform(no_tonic_data[subject_id][test_param + 2]))
    tonic_list.append('No tonic pain')
    phasic_list.append('High pain')
    vigour_list.append(transform(tonic_data[subject_id][test_param + 2]))
    tonic_list.append('With tonic pain')
    phasic_list.append('High pain')

    subject_list += [subject_id] * 6

tonic_all = np.array(tonic_nopain_list_vigour) + np.array(tonic_lowpain_list_vigour) + np.array(tonic_highpain_list_vigour)
notonic_all = np.array(notonic_nopain_list_vigour) + np.array(notonic_lowpain_list_vigour) + np.array(notonic_highpain_list_vigour)

res = stats.ttest_rel(tonic_all, notonic_all, alternative='two-sided')
print(res)

analyse_pd = pd.DataFrame({"Tonic": tonic_list, 
                            "Phasic": phasic_list,
                            "Vigour": vigour_list,
                            "Subjects": subject_list })

with localconverter(ro.default_converter + pandas2ri.converter):
  r_from_pd_df = ro.conversion.py2rpy(analyse_pd)

fml = ro.Formula("Vigour ~ Tonic * Phasic + Error(Subjects / (Tonic * Phasic))")
aov = rstatix.anova_test(formula=fml, data=r_from_pd_df)
print(aov)

analyse_pd.to_csv('temp/rate_no_exclusion')

import os
os.system("rscript vigour_repeated_anova" + EXPT_NAME + ".r")

ax = ax = axs[1]

ax.text(-0.15, 1.1, 'b', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')

pal_name = 'Set2'
pal = { "No tonic pain":  sns.color_palette(pal_name)[0], "With tonic pain": sns.color_palette(pal_name)[1]}
dx = 'Phasic'; dy = 'Vigour'; dhue = 'Tonic'; sigma = .2; ort = 'v'
pt.RainCloud(x = dx, y = dy, hue = dhue, data = analyse_pd, 
     palette = pal, bw = sigma, width_viol = .5, ax = ax,
     orient = ort , alpha = .65, dodge = True, pointplot = True,
     move = .2)

ax.tick_params(axis='both', labelsize=24)
ax.set_xlabel('Phasic pain conditions', fontsize=24)
ax.set_ylabel('', fontsize=24)
for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))

no_tonic_patch = mpatches.Patch(color=sns.color_palette('Set2')[0], label='No tonic pain')
tonic_patch = mpatches.Patch(color=sns.color_palette('Set2')[1], label='With tonic pain')
ax.legend(handles=[no_tonic_patch, tonic_patch], bbox_to_anchor=(1, 1.2), fontsize=24)

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=0, trim=True, ax=ax)

plt.tight_layout()
plt.savefig('figures/PUB/Expt4_vigour_fit.svg')