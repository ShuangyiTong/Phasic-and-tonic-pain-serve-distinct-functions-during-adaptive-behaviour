# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series

from core.individual_subject import get_end_of_trial_pain_ratings
from core.individual_subject import apply_corrections_natural_number_indexing
from core.individual_subject import get_series_from_control
from core.plot import add_significance_bar_hue, significance_converter

from core.experiment_data import set_expt

import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

EXPT_NAME = 'Expt4'
set_expt(EXPT_NAME)

pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]
exclude_participants = ['SUB20', 'SUB14']

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=exclude_participants, exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION])

subjects = list(exp_data.keys())
print(subjects)

start_trial_for_analysis = 6
end_trial_for_analysis = 24

all_ratings = get_multiple_series(exp_data, lambda individual_data: apply_corrections_natural_number_indexing(individual_data, get_end_of_trial_pain_ratings(individual_data), 'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis], subjects)
all_pain_conds = get_multiple_series(exp_data, lambda individual_data: list(map(
    lambda msg: msg.split('-')[-1], 
    get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], subjects)
print(all_ratings)

pain_rating_by_cond = [[(lambda r: sum(r) / len(r))([rating for rating, pain_cond in zip(ratings, pain_conds) if pain_cond == target_cond])
                            for ratings, pain_conds in zip(all_ratings, all_pain_conds)] 
                                for target_cond in pain_cond_idx]

plain_analyse_pd = pd.DataFrame({"Tonic": ['No tonic pain' if k.endswith('NoPressure') else 'With tonic pain' for k in pain_cond_idx for _ in subjects], 
                                "Phasic": ['No phasic pain' if k.startswith('NoPain') else ('Low phasic pain' if k.startswith('Low') else 'High phasic pain') for k in pain_cond_idx for _ in subjects],
                                "Rate": [r for k in pain_cond_idx for r in pain_rating_by_cond[pain_cond_idx.index(k)]],
                                "Subjects": [s for _ in pain_cond_idx for s in subjects] })

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

dx="Tonic"; dy="Rate"; ort="v"; pal_name="Set2"; sigma = .2
pal = { "No tonic pain":  sns.color_palette(pal_name)[0], "With tonic pain": sns.color_palette(pal_name)[1], "Difference": sns.color_palette(pal_name)[2] }

f = plt.figure(figsize=(21, 10))

axs = f.subplots(1, 2, sharey=True)
axs[0].set_ylim(-2, 10)

def plot_tonic_vs_no_tonic(ax, phasic_pain_name):
    analyse_pd = plain_analyse_pd.loc[plain_analyse_pd['Phasic'] == phasic_pain_name]
    difference_pd = pd.DataFrame({"Tonic": ['Difference'] * len(subjects), "Phasic": [phasic_pain_name] * len(subjects), 
                                "Rate": analyse_pd.loc[analyse_pd['Tonic'] == 'No tonic pain']['Rate'].to_numpy() - analyse_pd.loc[analyse_pd['Tonic'] == 'With tonic pain']['Rate'].to_numpy(),
                                "Subjects": subjects})
    analyse_pd = pd.concat([analyse_pd, difference_pd])

    ax=pt.half_violinplot( x = dx, y = dy, data = analyse_pd, palette=pal,
        bw = 0.5, cut = 0.,scale = "area", width = 0.5, inner = None, 
        orient = ort, alpha = 0.8, linewidth=0, ax=ax)
    ax=sns.stripplot( x = dx, y = dy, data = analyse_pd, palette = pal,
    edgecolor = "white",size = 9, jitter = 1, zorder = 0, alpha=1, dodge=True,
    orient = ort, ax=ax)
    ax=sns.boxplot( x = dx, y = dy, data = analyse_pd, color = "black", palette=pal,
                    width = 0.25, zorder = 10, showcaps = True, showfliers=True, ax=ax,
                    whiskerprops = {'linewidth':2, "zorder":10},
                    saturation = 1, orient = ort)
    ax.set_xlabel('', fontsize=32)
    ax.set_ylabel('', fontsize=32)
    ax.axhline(y=0, linestyle='dashed')

    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))

    ax.tick_params(axis='both', labelsize=32)

    significance_bar_vertical_len = 0.2
    max_value = analyse_pd['Rate'].to_numpy()
    max_value = np.nanmax(max_value) + significance_bar_vertical_len
    pvalue = stats.ttest_rel(analyse_pd.loc[analyse_pd['Tonic'] == 'No tonic pain']['Rate'], analyse_pd.loc[analyse_pd['Tonic'] == 'With tonic pain']['Rate'], alternative="two-sided").pvalue
    add_significance_bar_hue(ax, 0, 1, max_value, significance_bar_vertical_len, significance_converter(pvalue, raw=True), pvalue, show_insignificance=True, text_font=32)

    ax.set_xlabel(phasic_pain_name)

    sns.set_theme()
    sns.set_style("ticks")
    sns.despine(offset=0, trim=True, ax=ax)

axs[0].text(-0.05, 1.15, 'a', transform=axs[0].transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')
plot_tonic_vs_no_tonic(axs[0], 'Low phasic pain')
axs[1].text(-0.05, 1.15, 'b', transform=axs[1].transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')
plot_tonic_vs_no_tonic(axs[1], 'High phasic pain')
axs[0].set_ylabel('VAS Pain Ratings', fontsize=32)

no_tonic_patch = mpatches.Patch(color=sns.color_palette(pal_name)[0], label='No tonic pain')
tonic_patch = mpatches.Patch(color=sns.color_palette(pal_name)[1], label='With tonic pain')
axs[1].legend(handles=[no_tonic_patch, tonic_patch], loc='upper right', fontsize=24)

plt.tight_layout()
plt.savefig('figures/PUB/Ratings_by_conditions_expt4.svg')

pain_pd = plain_analyse_pd.loc[(plain_analyse_pd['Phasic'] == 'Low phasic pain') | (plain_analyse_pd['Phasic'] == 'High phasic pain')]
pain_pd.to_csv('temp/rate_no_exclusion')

import os
os.system("rscript rate_repeated_anova" + EXPT_NAME + ".r")