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

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, LIVEAMP_DEVICE_NAME, EEGLAB_NAME, UNITY_DEVICE_NAME
from core.plot import add_significance_bar_hue_inverted, get_x_positions, significance_converter
from core.experiment_data import set_expt

EXPT_NAME = 'Expt4'
set_expt(EXPT_NAME)

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series

from core.individual_subject import get_end_of_trial_pain_ratings
from core.individual_subject import apply_corrections_natural_number_indexing
from core.individual_subject import get_all_collected_fruits
from core.individual_subject import get_series_from_control

from core.utils import save_cache, load_cache

import matplotlib.pyplot as plt

cache = None #load_cache('collection_bias' + EXPT_NAME)

pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]
start_trial_for_analysis = 6
end_trial_for_analysis = 24
if cache:
    exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=['SUB14', 'SUB20'], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME, UNITY_DEVICE_NAME])
else:   
    exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=['SUB14', 'SUB20'], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME])

subjects = list(exp_data.keys())

def sigmoid_pain(params, x):
    return params[2] / (1 + math.exp(-params[0] * (x - params[1])))

import json

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


f = plt.figure(figsize=(31, 10))

axs = f.subplots(1, 3, sharey=True)

ax = axs[0]

ax.text(-0.2, 1.1, 'a', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')

no_tonic_avg = []
tonic_avg = []
xs = np.arange(-0.2, 1.2, 0.01)
for no_tonic_data, tonic_data in zip(no_tonic_data_list, tonic_data_list):
    ys = [sigmoid_pain(([no_tonic_data[0], no_tonic_data[1], no_tonic_data[4]]), x) for x in xs]
    print(ys)
    ax.plot(xs, ys, c=sns.color_palette('Set2')[0], alpha=0.3)
    no_tonic_avg.append(ys)
    ys = [sigmoid_pain(([tonic_data[0], tonic_data[1], tonic_data[4]]), x) for x in xs]
    ax.plot(xs, ys, c=sns.color_palette('Set2')[1], alpha=0.3)
    tonic_avg.append(ys)

ax.plot(xs, np.mean(np.array(no_tonic_avg), axis=0), c=sns.color_palette('Set2')[0], label='No tonic pain', linewidth=3)
ax.plot(xs, np.mean(np.array(tonic_avg), axis=0), c=sns.color_palette('Set2')[1], label='With tonic pain', linewidth=3)

ax.invert_yaxis()

ax.tick_params(axis='both', labelsize=24)
ax.set_xlabel('Electric shock intensity', fontsize=32)
ax.set_ylabel('Phasic pain utility function $C_p u$', fontsize=32)

levels = [0, 0.5, 1]
ax.set_xticks(levels)
ax.set_xticklabels(["0%", "50%", "100%"])
for level in levels:
    ax.axvline(level, linestyle='dashed')

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=0, trim=True, ax=ax)

ax = axs[1]

ax.text(-0.1, 1.1, 'b', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')

pain_val_at_05_tonic = [sigmoid_pain((v[0], v[1], v[4]), 0.5) for v in tonic_data_list]
pain_val_at_05_no_tonic = [sigmoid_pain((v[0], v[1], v[4]), 0.5) for v in no_tonic_data_list]

test05_res = stats.ttest_rel(pain_val_at_05_tonic, pain_val_at_05_no_tonic, alternative='two-sided')
analyse_pd = pd.DataFrame({'Data': np.array(pain_val_at_05_no_tonic + pain_val_at_05_tonic), 
                           'Group': ['No tonic pain'] * len(pain_val_at_05_no_tonic) + ['With tonic pain'] * len(pain_val_at_05_tonic)})
dx = 'Group'
dy = 'Data'
pal='Set2'
ort = 'v'
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
significance_bar_vertical_len = 0.5
max_value = np.nanmin(max_value) - significance_bar_vertical_len * 2
add_significance_bar_hue_inverted(ax, 0, 1, max_value, significance_bar_vertical_len, significance_converter(test05_res.pvalue), test05_res.pvalue, show_insignificance=True)

print(np.mean(pain_val_at_05_no_tonic), np.std(pain_val_at_05_no_tonic))
print(np.mean(pain_val_at_05_tonic), np.std(pain_val_at_05_tonic))
print(test05_res)

ax.tick_params(axis='both', labelsize=32)
ax.set_xlabel('', fontsize=32)
ax.set_ylabel('', fontsize=32)
ax.tick_params(labelleft=False, left=False)

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=0, trim=True, left=True, ax=ax)

if not cache:
    all_pain_conds = get_multiple_series(exp_data, lambda individual_data: list(map(
                        lambda msg: msg.split('-')[-1], 
                        get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], subjects)

    all_collection_bias = get_multiple_series(exp_data, lambda individual_data: [sum(map(lambda x : x.endswith('G'), fruits)) / len(fruits) 
                                                                                    for fruits in get_all_collected_fruits(individual_data)][start_trial_for_analysis:end_trial_for_analysis], subjects)
    save_cache((all_pain_conds, all_collection_bias), 'collection_bias' + EXPT_NAME)
else:
    all_pain_conds, all_collection_bias = cache

ax = axs[2]

ax.text(-0.1, 1.1, 'c', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')

collection_bias_by_conds = [[(lambda r: np.nanmean(r))([moving_speed for moving_speed, pain_cond in zip(moving_speeds, pain_conds) if pain_cond == target_cond])
                            for moving_speeds, pain_conds in zip(all_collection_bias, all_pain_conds)] 
                                for target_cond in pain_cond_idx]

no_tonic_average = (np.array(collection_bias_by_conds)[0] + 
                    np.array(collection_bias_by_conds)[2] + 
                    np.array(collection_bias_by_conds)[4]) / 3
tonic_average = (np.array(collection_bias_by_conds)[1] + 
        np.array(collection_bias_by_conds)[3] + 
        np.array(collection_bias_by_conds)[5]) / 3

import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter
import rpy2.robjects.lib.ggplot2 as ggplot2

base = importr("base")
rstats = importr("stats")
lme4 = importr('lme4')
lmer_test = importr('lmerTest')

analyse_pd = pd.DataFrame({'Tonic_C_pu': pain_val_at_05_tonic, 'bias': tonic_average})
sns.regplot(x='bias', y='Tonic_C_pu', data=analyse_pd, ax=ax, scatter_kws={'s': 48}, color=sns.color_palette('Set2')[1])

with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(analyse_pd)

fml = ro.Formula('Tonic_C_pu ~ bias')
lm = rstats.lm(fml, data=r_from_pd_df)
print(base.summary(lm))
print('END OF TONIC =====================================================')

analyse_pd = pd.DataFrame({'No_tonic_C_pu': pain_val_at_05_no_tonic, 'bias': no_tonic_average})
sns.regplot(x='bias', y='No_tonic_C_pu', data=analyse_pd, ax=ax, scatter_kws={'s': 48}, color=sns.color_palette('Set2')[0])

with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(analyse_pd)

fml = ro.Formula('No_tonic_C_pu ~ bias')
lm = rstats.lm(fml, data=r_from_pd_df)
print(base.summary(lm))
print('END OF NO TONIC =====================================================')

ax.tick_params(axis='both', labelsize=32)
ax.set_xlabel('Aversive choice probability', fontsize=32)
ax.set_ylabel('', fontsize=32)
ax.tick_params(labelleft=False, left=False)
ax.set_xlim((0.1, 0.6))

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=0, trim=True, left=True, ax=ax)

plt.tight_layout(pad=3)

transFigure = f.transFigure.inverted()
for i, pain_func_v in enumerate(pain_val_at_05_tonic):
    coord1 = transFigure.transform(axs[0].transData.transform([0.5, pain_func_v]))
    coord2 = transFigure.transform(axs[2].transData.transform([tonic_average[i], pain_func_v]))
    line = mpl.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=f.transFigure, alpha=0.05, color=sns.color_palette('Set2')[1])
    f.lines.append(line)

for i, pain_func_v in enumerate(pain_val_at_05_no_tonic):
    coord1 = transFigure.transform(axs[0].transData.transform([0.5, pain_func_v]))
    coord2 = transFigure.transform(axs[2].transData.transform([no_tonic_average[i], pain_func_v]))
    line = mpl.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=f.transFigure, alpha=0.05, color=sns.color_palette('Set2')[0])
    f.lines.append(line)

no_tonic_patch = mpatches.Patch(color=sns.color_palette('Set2')[0], label='No tonic pain')
tonic_patch = mpatches.Patch(color=sns.color_palette('Set2')[1], label='With tonic pain')
ax.legend(handles=[no_tonic_patch, tonic_patch], bbox_to_anchor=(0.89, 1.15), fontsize=24)

plt.savefig('figures/PUB/Expt4_phasic_fitting.svg')