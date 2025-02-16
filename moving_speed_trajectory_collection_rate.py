# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import os

from core.experiment_data import get_multiple_series, make_experiment_data, set_expt
from core.individual_subject import get_series_from_control
from core.utils import load_cache
from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME
from core.plot import add_significance_bar_hue, get_x_positions, significance_converter

import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import pingouin as pg

set_expt('Expt4')
exp_data = make_experiment_data(exclude_participants=['SUB14', 'SUB20'], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME])
subjects_expt4 = list(exp_data.keys())
start_trial_for_analysis_expt4 = 6
end_trial_for_analysis_expt4 = 24
all_pain_conds_expt4 = get_multiple_series(exp_data, lambda individual_data: list(map(
    lambda msg: msg.split('-')[-1], 
    get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis_expt4:end_trial_for_analysis_expt4], subjects_expt4)

os.system('Python moving_speed.py Expt4')

pain_cond_idx_expt4 = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]
moving_speed_expt4 = load_cache('moving rate' + 'Expt4TotalHand3dDist')
moving_speed_by_cond = [[(lambda r: np.nanmean(r))([moving_speed for moving_speed, pain_cond in zip(moving_speeds, pain_conds) if pain_cond == target_cond])
                            for moving_speeds, pain_conds in zip(moving_speed_expt4, all_pain_conds_expt4)] 
                                for target_cond in pain_cond_idx_expt4]


f = plt.figure(figsize=(32, 22))

from matplotlib.gridspec import GridSpec

sps1, sps2, _, _ = GridSpec(2, 2)
ax = f.add_subplot(2, 2, 1)

no_tonic_average = (np.array(moving_speed_by_cond)[0] + 
                    np.array(moving_speed_by_cond)[2] + 
                    np.array(moving_speed_by_cond)[4]) / 3
tonic_average = (np.array(moving_speed_by_cond)[1] + 
        np.array(moving_speed_by_cond)[3] + 
        np.array(moving_speed_by_cond)[5]) / 3

plain_analyse_pd = pd.DataFrame({'Data': no_tonic_average.tolist() + tonic_average.tolist(), # + (no_tonic_average - tonic_average).tolist(), 
                                 'Group': ['No tonic pain'] * len(no_tonic_average) + ['With tonic pain'] * len(tonic_average)})# + ['Difference'] * len(tonic_average)})
res = stats.ttest_rel(no_tonic_average, tonic_average, alternative='greater')

dx = 'Group'
dy = 'Data'
ort = 'v'
pal = 'Set2'

ax.text(-0.1, 1.1, 'a', transform=ax.transAxes,
    fontsize=48, fontweight='bold', va='top', ha='right')

ax=pt.half_violinplot( x = dx, y = dy, data = plain_analyse_pd, palette=pal,
    bw = 0.5, cut = 0.,scale = "area", width = 0.5, inner = None, 
    orient = ort, alpha = 0.8, linewidth=0, ax=ax)

ax=sns.stripplot( x = dx, y = dy, data = plain_analyse_pd, palette=pal,
    edgecolor = "white",size = 6, jitter = 1, zorder = 0, 
    orient = ort, ax=ax)

ax=sns.boxplot( x = dx, y = dy, data = plain_analyse_pd, palette=pal,
    width = .25, zorder = 10, showcaps = True, showfliers=True,
    whiskerprops = {'linewidth':2, "zorder":10}, 
    saturation = 1, orient = ort, ax=ax)

for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))

ax.set_xlabel('')
ax.tick_params(axis='both', labelsize=32)
# ax.axhline(0, linestyle='dashed')

max_value = plain_analyse_pd['Data'].to_numpy()
significance_bar_vertical_len = 0.025
max_value = np.nanmax(max_value) + significance_bar_vertical_len
print('moving speed: ', res)
diff = no_tonic_average - tonic_average
print('Cohen d=', np.mean(diff) / np.std(diff, ddof=1))
bf10 = pg.bayesfactor_ttest(res.statistic, 31, paired=True)
print("BF_10=", bf10)
print('no tonic moving speed M: ', np.mean(no_tonic_average), 'SD: ', np.std(no_tonic_average))
print('tonic moving speed M:', np.mean(tonic_average), 'SD: ', np.std(tonic_average))
print('pair wise M: ', np.mean(no_tonic_average - tonic_average), 'SD: ', np.std(no_tonic_average - tonic_average))
ax.set_ylabel('Hand speed (m/s)', fontsize=32)
add_significance_bar_hue(ax, 0, 1, max_value, significance_bar_vertical_len, significance_converter(res.pvalue), res.pvalue, text_font=36)

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=0, trim=True, ax=ax)

ax = f.add_subplot(2, 1, 2)

ax.text(-0.05, 1.1, 'c', transform=ax.transAxes,
    fontsize=48, fontweight='bold', va='top', ha='right')

import hand_reaching_trajectory

reaching_trajectory_tonic = np.array(hand_reaching_trajectory.reaching_trajectory_tonic)
reaching_trajectory_no_tonic = np.array(hand_reaching_trajectory.reaching_trajectory_no_tonic)

difference_matrix = []
for i in range(len(hand_reaching_trajectory.subjects)):
    # ax.plot(np.nanmean(np.array(reaching_trajectory_no_tonic[i]), axis=0) - np.nanmean(np.array(reaching_trajectory_tonic[i]), axis=0), c=sns.color_palette(pal)[2], alpha=0.1)
    difference_matrix.append(np.nanmean(np.array(reaching_trajectory_no_tonic[i]), axis=0) - np.nanmean(np.array(reaching_trajectory_tonic[i]), axis=0))
difference_matrix = np.array(difference_matrix)
ax.axhline(y=0, color='black', linestyle='dashed')
ax.plot(np.nanmean(reaching_trajectory_tonic, axis=(0, 1)) * 1000, c=sns.color_palette(pal)[1], linewidth=5)
ax.plot(np.nanmean(reaching_trajectory_no_tonic, axis=(0, 1)) * 1000, c=sns.color_palette(pal)[0], linewidth=5)
ax.plot(np.mean(difference_matrix, axis=0) * 1000, c=sns.color_palette(pal)[2])
ax.fill_between(np.arange(np.array(difference_matrix).shape[-1]),
                 np.mean(difference_matrix, axis=0) * 1000 - 1.96 * stats.sem(difference_matrix, axis=0) * 1000,
                 np.mean(difference_matrix, axis=0) * 1000 + 1.96 * stats.sem(difference_matrix, axis=0) * 1000, alpha=0.5, color=sns.color_palette(pal)[2]), 

ax.tick_params(axis='both', labelsize=24)
ax.set_ylabel('Instantaneous hand speed (m/s)', fontsize=32)
ax.set_xticks([2 * i for i in range(11)])
ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])
ax.grid()
ax.set_xlabel('Distance travelled', fontsize=32)

no_tonic_patch = mpatches.Patch(color=sns.color_palette(pal)[0], label='No tonic pain')
tonic_patch = mpatches.Patch(color=sns.color_palette(pal)[1], label='With tonic pain')
ax.legend(handles=[no_tonic_patch, tonic_patch], bbox_to_anchor=(0.5, 1), fontsize=24)


sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=0, trim=True, ax=ax)

ax = f.add_subplot(2, 2, 2)

os.system('Python collection_rate.py')

pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]
all_collection_rates = load_cache('collection rateExpt4')

collection_rates_by_cond = [[(lambda r: np.nanmean(r))([collection_rate for collection_rate, pain_cond in zip(collection_rates, pain_conds) if pain_cond == target_cond])
                            for collection_rates, pain_conds in zip(all_collection_rates, all_pain_conds_expt4)] 
                                for target_cond in pain_cond_idx]

no_tonic_average = (np.array(collection_rates_by_cond)[0] + 
                                            np.array(collection_rates_by_cond)[2] + 
                                            np.array(collection_rates_by_cond)[4]) / 3
tonic_average = (np.array(collection_rates_by_cond)[1] + 
        np.array(collection_rates_by_cond)[3] + 
        np.array(collection_rates_by_cond)[5]) / 3 

plain_analyse_pd = pd.DataFrame({'Data': no_tonic_average.tolist() + tonic_average.tolist(), # + (no_tonic_average - tonic_average).tolist(), 
                                 'Group': ['No tonic pain'] * len(no_tonic_average) + ['With tonic pain'] * len(tonic_average)})# + ['Difference'] * len(tonic_average)})
res = stats.ttest_rel(no_tonic_average, tonic_average, alternative='greater')

dx = 'Group'
dy = 'Data'

ax.text(-0.05, 1.1, 'b', transform=ax.transAxes,
    fontsize=48, fontweight='bold', va='top', ha='right')

ax=pt.half_violinplot( x = dx, y = dy, data = plain_analyse_pd, palette=pal,
    bw = 0.5, cut = 0.,scale = "area", width = 0.5, inner = None, 
    orient = ort, alpha = 0.8, linewidth=0, ax=ax)

ax=sns.stripplot( x = dx, y = dy, data = plain_analyse_pd, palette=pal,
    edgecolor = "white",size = 6, jitter = 1, zorder = 0, 
    orient = ort, ax=ax)

ax=sns.boxplot( x = dx, y = dy, data = plain_analyse_pd, palette=pal,
    width = .25, zorder = 10, showcaps = True, showfliers=True,
    whiskerprops = {'linewidth':2, "zorder":10}, 
    saturation = 1, orient = ort, ax=ax)

for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))

ax.set_xlabel('')
ax.tick_params(axis='both', labelsize=32)
# ax.axhline(0, linestyle='dashed')

max_value = plain_analyse_pd['Data'].to_numpy()
significance_bar_vertical_len *= 10
max_value = np.nanmax(max_value) + significance_bar_vertical_len
print('collection rate: ', res)
diff = no_tonic_average - tonic_average
print('Cohen d=', np.mean(diff) / np.std(diff, ddof=1))
bf10 = pg.bayesfactor_ttest(res.statistic, 31, paired=True)
print("BF_10=", bf10)
print('no tonic collection rate M: ', np.mean(no_tonic_average), 'SD: ', np.std(no_tonic_average))
print('tonic collection M:', np.mean(tonic_average), 'SD: ', np.std(tonic_average))
print('pair wise M :', np.mean(no_tonic_average - tonic_average), 'SD: ', np.std(no_tonic_average - tonic_average))
ax.set_ylabel('Collection rate per block (1 minute)', fontsize=32)
add_significance_bar_hue(ax, 0, 1, max_value, significance_bar_vertical_len, significance_converter(res.pvalue), res.pvalue, text_font=32)

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=0, trim=True, ax=ax)

plt.tight_layout(pad=3)
plt.savefig('figures/PUB/vigour_behavioural_plot.svg')
