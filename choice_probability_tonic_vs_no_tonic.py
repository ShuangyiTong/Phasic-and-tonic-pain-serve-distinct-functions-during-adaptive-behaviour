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

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, LIVEAMP_DEVICE_NAME, EEGLAB_NAME
from core.plot import add_significance_bar_hue, get_x_positions, significance_converter
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

pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]
start_trial_for_analysis = 6
end_trial_for_analysis = 24
exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=['SUB14', 'SUB20'], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME])

subjects = list(exp_data.keys())

f = plt.figure(figsize=(21, 6))

axs = f.subplots(1, 3)

all_collection_bias = get_multiple_series(exp_data, lambda individual_data: [sum(map(lambda x : x.endswith('G'), fruits)) / len(fruits) 
                                                                                for fruits in get_all_collected_fruits(individual_data)][start_trial_for_analysis:end_trial_for_analysis], subjects)
                            
all_pain_conds = get_multiple_series(exp_data, lambda individual_data: list(map(
                    lambda msg: msg.split('-')[-1], 
                    get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], subjects)

all_pain_ratings = get_multiple_series(exp_data, lambda individual_data: apply_corrections_natural_number_indexing(
                                                    individual_data, get_end_of_trial_pain_ratings(individual_data), 
                                                    'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis], subjects)

collection_bias_by_conds = [[(lambda r: np.nanmean(r))([collection_bias for collection_bias, pain_cond in zip(collection_biases, pain_conds) if pain_cond == target_cond])
                            for collection_biases, pain_conds in zip(all_collection_bias, all_pain_conds)] 
                                for target_cond in pain_cond_idx]

dicts = [[{ 'choice_prob' : bias, 'participant' : participant_index, 'rating' : pain_rating, 'cond': pain_cond} for pain_rating, bias, pain_cond in zip(pain_ratings, bias, pain_conds)] 
    for participant_index, pain_ratings, bias, pain_conds in zip(subjects, all_pain_ratings, all_collection_bias, all_pain_conds)]
save_cache(dicts, 'choice_bias')
import numpy as np
import pandas as pd

analyse_pd = pd.DataFrame()
for participant in dicts:
    for block in participant:
        analyse_pd = analyse_pd.append({ k: v for k, v in block.items()}, ignore_index=True)

analyse_pd = analyse_pd[analyse_pd['choice_prob'].notna()]

print(analyse_pd.to_string())

import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

base = importr("base")
lme4 = importr('lme4')
lmer_test = importr('lmerTest')
rstats = importr('stats')
broom = importr('broom.mixed')
generics = importr('generics')
import rpy2.robjects.lib.ggplot2 as ggplot2

for selective_cond in ['NoPressure', 'Tonic']:
    print('>>>>>>>>>>>>>>>>>>>>> RESULT FOR ' + selective_cond)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(analyse_pd.loc[analyse_pd['cond'].str.contains(selective_cond)])
    m = lmer_test.lmer('choice_prob ~ rating + (1 + rating | participant)', r_from_pd_df)

    print(base.summary(m))
    print(rstats.confint(m))
    print(generics.tidy(m))
    print(rstats.anova(m))

import matplotlib.colors as cl
import matplotlib.cm as cm
import seaborn as sns
import matplotlib as mpl
sns.set_theme()
sns.set_style("ticks")
cmap = mpl.cm.get_cmap('Spectral')


from core.plot import all_curves

def plot_all_subjects_pain_value_curve(curves, intercept, slope, cbar_start=0.23):
    curve_val = [[] for i in range(11)]
    norm = mpl.colors.Normalize(vmin=0.25, vmax=0.65)
    for curve in curves:
        for r, c in zip(curve[0], curve[1]):
            if c == c:
                curve_val[r].append(c)
        
        new_curve_r = [r for r, c in zip(curve[0], curve[1]) if c == c]
        new_curve_c = [c for c in curve[1] if c == c]
        print(np.mean(new_curve_c))
        ax.scatter(new_curve_r, new_curve_c, color=cmap(norm(new_curve_c[0])), alpha=0.7)
        ax.plot(new_curve_r, new_curve_c, color=cmap(norm(new_curve_c[0])), alpha=0.4)

    ax.set_ylabel("Aversive Choice Probability", fontsize=24)
    ax.plot([0, 10], [intercept, slope * 10 + intercept], color='black', linestyle='dashed', label='Linear mixed model fitted line\n (p=0.0734, t=1.866, df=26.012)')

    ax.tick_params(axis='both', labelsize=24)
    sns.despine(offset=0, trim=True, ax=ax)
    cbar_ax = f.add_axes([cbar_start, 0.90, 0.08, 0.02])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cbar_ax.set_title('At VAS=0', fontsize=18)

def keep_condition_with_string(condition, all_lst):
    return list(map(lambda individual_pair: list(map(lambda filtered_pair: filtered_pair[1], 
                                                     list(filter(lambda single_block_pair: condition in single_block_pair[0], zip(individual_pair[0], individual_pair[1])))))
                    ,zip(all_pain_conds, all_lst)))


ax = axs[0]

ax.text(-0.15, 1.1, 'a', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')

# lmer result with bias ~ rating + (1 + rating | participant), check this agrees with the output from R
intercept = 0.472614
slope = -0.028590
curves = all_curves(keep_condition_with_string('NoPressure', all_pain_ratings), keep_condition_with_string('NoPressure', all_collection_bias))
plot_all_subjects_pain_value_curve(curves, intercept, slope, 0.23)
ax.set_xlabel("No tonic pain: VAS Pain Ratings", fontsize=24)

ax = axs[1]

intercept = 0.472404
slope = -0.027969
ax.text(-0.15, 1.1, 'b', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')
curves = all_curves(keep_condition_with_string('Tonic', all_pain_ratings), keep_condition_with_string('Tonic', all_collection_bias))
plot_all_subjects_pain_value_curve(curves, intercept, slope, 0.53)
ax.set_xlabel("With tonic pain: VAS Pain Ratings", fontsize=24)

plain_analyse_pd = pd.DataFrame({"Tonic": ['No tonic pain' if k.endswith('NoPressure') else 'With tonic pain' for k in pain_cond_idx for _ in subjects], 
                                "Phasic": ['No phasic pain' if k.startswith('NoPain') else ('Low phasic pain' if k.startswith('Low') else 'High phasic pain') for k in pain_cond_idx for _ in subjects],
                                "Rate": [r for k in pain_cond_idx for r in collection_bias_by_conds[pain_cond_idx.index(k)]],
                                "Subjects": [s for _ in pain_cond_idx for s in subjects] })

import os

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

base = importr("base")
rstats = importr("stats")
rstatix = importr("rstatix")

with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(plain_analyse_pd)


fml = ro.Formula("Rate ~ Tonic * Phasic + Error(Subjects / (Tonic * Phasic))")
aov = rstatix.anova_test(formula=fml, data=r_from_pd_df)
print(aov)

plain_analyse_pd.to_csv('temp/rate_no_exclusion')

import os
os.system("rscript rate_repeated_anova" + EXPT_NAME + ".r")

ax = axs[2]

no_tonic_average = (np.array(collection_bias_by_conds)[0] + 
                    np.array(collection_bias_by_conds)[2] + 
                    np.array(collection_bias_by_conds)[4]) / 3
tonic_average = (np.array(collection_bias_by_conds)[1] + 
        np.array(collection_bias_by_conds)[3] + 
        np.array(collection_bias_by_conds)[5]) / 3

plot_pd = pd.DataFrame({'Data': no_tonic_average.tolist() + tonic_average.tolist() + (no_tonic_average - tonic_average).tolist(), 
                        'Group': ['No tonic\n pain'] * len(no_tonic_average) + ['With tonic\n pain'] * len(tonic_average) + ['Difference'] * len(tonic_average)})

res = stats.ttest_rel(no_tonic_average, tonic_average, alternative='two-sided')

dx = 'Group'
dy = 'Data'
ort = 'v'
pal = 'Set2'

ax.text(-0.15, 1.1, 'c', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')

ax=pt.half_violinplot( x = dx, y = dy, data = plot_pd, palette=pal,
    bw = 0.5, cut = 0.,scale = "area", width = 0.5, inner = None, 
    orient = ort, alpha = 0.8, linewidth=0, ax=ax)

ax=sns.stripplot( x = dx, y = dy, data = plot_pd, palette=pal,
    edgecolor = "white",size = 6, jitter = 1, zorder = 0, 
    orient = ort, ax=ax)

ax=sns.boxplot( x = dx, y = dy, data = plot_pd, palette=pal,
    width = .25, zorder = 10, showcaps = True, showfliers=True,
    whiskerprops = {'linewidth':2, "zorder":10}, 
    saturation = 1, orient = ort, ax=ax)

for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))


no_tonic_patch = mpatches.Patch(color=sns.color_palette(pal)[0], label='No tonic pain')
tonic_patch = mpatches.Patch(color=sns.color_palette(pal)[1], label='With tonic pain')
ax.legend(handles=[no_tonic_patch, tonic_patch], bbox_to_anchor=(0.65, 1.05), fontsize=18)

ax.set_xlabel('')
ax.tick_params(axis='both', labelsize=24)

max_value = plot_pd['Data'].to_numpy()
significance_bar_vertical_len = 0.02
max_value = np.nanmax(max_value) + significance_bar_vertical_len
print('collection bias: ', res)
print('no tonic collection bias M: ', np.mean(no_tonic_average), 'SD: ', np.std(no_tonic_average))
print('tonic collection bias M:', np.mean(tonic_average), 'SD: ', np.std(tonic_average))
ax.set_ylabel('Aversive choice probability', fontsize=24)
add_significance_bar_hue(ax, 0, 1, max_value, significance_bar_vertical_len, significance_converter(res.pvalue), res.pvalue, show_insignificance=True)

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=0, trim=True, ax=ax)

plt.tight_layout()
plt.savefig('figures/PUB/choice_probability_tonic.svg')