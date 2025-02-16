# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import sys

Expt_name = 'Expt2'

import dill

from core.utils import nan_square_sum
from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_NAME, ARDUINO_DEVICE_NAME

import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

from core.experiment_data import make_experiment_data, get_multiple_series, set_expt
from core.individual_subject import apply_corrections_natural_number_indexing, get_end_of_trial_pain_ratings, get_series_from_control

set_expt('Expt2')
exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB11"], exclude_device_data=[UNITY_DEVICE_NAME, NI_DEVICE_NAME, ARDUINO_DEVICE_NAME])

subjects = list(exp_data.keys())
start_trial_for_analysis = -10
end_trial_for_analysis = None
    
pain_cond_idx = [ "NoPain", "MidLowPain", "MidMidPain", "MidHighPain", "MaxPain" ]

all_ratings = get_multiple_series(exp_data, lambda individual_data: apply_corrections_natural_number_indexing(individual_data, get_end_of_trial_pain_ratings(individual_data), 'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis], subjects)
all_pain_conds = get_multiple_series(exp_data, lambda individual_data: list(map(
    lambda msg: msg.split('-')[-1], 
    get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], subjects)
pain_rating_by_conds = [(lambda r: sum(r) / len(r))([rating for rating, pain_cond in zip(ratings, pain_conds) if pain_cond == target_cond])
                            for target_cond in pain_cond_idx for ratings, pain_conds in zip(all_ratings, all_pain_conds)]

stimulation_result_file_name = 'temp/' + Expt_name + '_fit_fin.dill'
acc = []

all_stimulation_results = None
with open(stimulation_result_file_name, 'rb') as f:
    all_stimulation_results = dill.load(f)

results = {}
detailed_results = {}
results_index = 1
analyse_pd = pd.DataFrame()

def get_pain_scale(positivity, horizontal, vertical):
    return positivity * (1 - math.sqrt(nan_square_sum([horizontal, vertical])))

def get_pain_func(param1, param2, positivity, horizontal, vertical):
    scale = get_pain_scale(positivity, horizontal, vertical)
    return lambda x: scale / (1 + math.exp(-param1 * (x - param2)))

for subject_name, stimulation_results in zip(all_stimulation_results[0], all_stimulation_results[1]):
    if subject_name == 'SUB11': # TODO: remove this to model fitting
        continue
    
    results[subject_name] = list(stimulation_results[results_index][0])
    print(subject_name)
    print(results[subject_name], stimulation_results[results_index][1])
    acc.append(stimulation_results[results_index][1])
    pain_func = get_pain_func(stimulation_results[results_index][0][0],
                            stimulation_results[results_index][0][1],
                            stimulation_results[results_index][0][2],
                            stimulation_results[results_index][0][3],
                            stimulation_results[results_index][0][4])
    subject_dict_pain_scale_at_max = {'Name': subject_name, 
                    'PainParam1': stimulation_results[results_index][0][0],
                    'PainParam2': stimulation_results[results_index][0][1],
                    'PainPositivity': stimulation_results[results_index][0][2],
                    'Horizontal': stimulation_results[results_index][0][3],
                    'Vertical': stimulation_results[results_index][0][4],
                    'PainScale': get_pain_scale(stimulation_results[results_index][0][2],
                                                stimulation_results[results_index][0][3],
                                                stimulation_results[results_index][0][4]),
                    'PainValueAtMax': pain_func(1)}
    analyse_pd = analyse_pd.append(subject_dict_pain_scale_at_max, ignore_index=True)

print(analyse_pd)
print("Mean accuracy:", np.mean(acc))

ttest_res = stats.ttest_1samp(analyse_pd['PainScale'].tolist(), popmean=0, alternative='two-sided')
print("Pain < 0:", ttest_res, "Mean: ", np.mean(analyse_pd['PainScale']), "SD: ", np.std(analyse_pd['PainScale']))

dx = "PainScale"
dy = None
ort = "h"
pal = sns.color_palette("husl", 9)

f = plt.figure(figsize=(21, 10))
ax = f.add_subplot(1, 2, 1)
ax.text(0, 1.15, 'a', transform=ax.transAxes,
    fontsize=42, fontweight='bold', va='top', ha='right')

ax=pt.half_violinplot( x = dx, y = dy, data = analyse_pd, color=pal[0],
      bw = 0.2, cut = 0.,scale = "area", width = .7, inner = None, 
      orient = ort, alpha = 1, linewidth=0)

ax=sns.stripplot( x = dx, y = dy, data = analyse_pd, color=pal[0],
     edgecolor = "white",size = 10, jitter = 1, zorder = 0,  alpha=1,
     orient = ort)

ax=sns.boxplot( x = dx, y = dy, data = analyse_pd, color = "black",
      width = .15, zorder = 10, showcaps = True,
      boxprops = {'facecolor':'none', "zorder":10}, showfliers=True,
      whiskerprops = {'linewidth':2, "zorder":10}, 
      saturation = 1, orient = ort)

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=-200, left=True, trim=True)
ax.set_yticks([])
# ax.set_xticks([-1 + 0.2 * i for i in range(9)])
ax.set_xlabel('Fitted phasic pain utility function coefficient $C_p$', fontsize=30)

ax.tick_params(axis='both', labelsize=24)

plt.tight_layout()

dx = 'Horizontal'
pal = sns.color_palette("Set2")

ax = f.add_subplot(1, 2, 2)
ax.text(0, 1.15, 'b', transform=ax.transAxes,
    fontsize=42, fontweight='bold', va='top', ha='right')

ax=pt.half_violinplot( x = dx, y = dy, data = analyse_pd, color=pal[0],
      bw = 0.2, cut = 0.,scale = "area", width = .6, inner = None, 
      orient = ort, alpha = 0.8, linewidth=0)

ax=sns.stripplot( x = dx, y = dy, data = analyse_pd, color=pal[0],
     edgecolor = "white",size = 10, jitter = 1, zorder = 0, 
     orient = ort)

ax=sns.boxplot( x = dx, y = dy, data = analyse_pd, color = "black",
      width = .15, zorder = 10, showcaps = True,
      boxprops = {'facecolor': list(pal[0]) + [0.4], "zorder":10}, showfliers=True,
      whiskerprops = {'linewidth':2, "zorder":10}, 
      saturation = 1, orient = ort)

dx = 'Vertical'

ax=pt.half_violinplot( x = dx, y = dy, data = analyse_pd, color=pal[1],
      bw = 0.2, cut = 0.,scale = "area", width = .6, inner = None, 
      orient = ort, alpha = 0.8, linewidth=0)

ax=sns.stripplot( x = dx, y = dy, data = analyse_pd, color=pal[1],
     edgecolor = "white",size = 10, jitter = 1, zorder = 0, 
     orient = ort)

ax=sns.boxplot( x = dx, y = dy, data = analyse_pd, color = "black",
      width = .15, zorder = 10, showcaps = True,
      boxprops = {'facecolor': list(pal[1]) + [0.4], "zorder":10}, showfliers=True,
      whiskerprops = {'linewidth':2, "zorder":10}, 
      saturation = 1, orient = ort)

ax.set_xlabel('Fitted moving effort coefficient $C_m$\n in separate horizontal and vertical components', fontsize=30)
ax.set_ylabel('')

horizontal_patch = mpatches.Patch(color=pal[0], label='Horizontal component')
vertical_patch = mpatches.Patch(color=pal[1], label='Vertical component')
ax.legend(handles=[horizontal_patch, vertical_patch], loc='upper left', fontsize=18)

sns.set_theme()
sns.set_style("ticks")
sns.despine(offset=-200, left=True, trim=True)
ax.set_yticks([])

ax.tick_params(axis='both', labelsize=24)

ttest_res = stats.ttest_rel(analyse_pd['Horizontal'], analyse_pd['Vertical'], alternative='two-sided')
print("Horizontal > Vertical:", ttest_res, "Mean: ", np.mean(analyse_pd['Horizontal']), "SD: ", np.std(analyse_pd['Horizontal']),
      "Vertical: ", np.mean(analyse_pd['Vertical']), "SD: ", np.std(analyse_pd['Vertical']))

plt.tight_layout(rect=(0, -0.15, 1, 0.98))

plt.savefig('figures/PUB/Expt2_Model_Fitting_Coefficients.svg')
plt.clf()
plt.close()

fig = plt.figure(figsize=(31, 10))

ax = fig.add_subplot(1, 3, 1)

ax.text(-0.1, 1.15, 'c', transform=ax.transAxes,
    fontsize=42, fontweight='bold', va='top', ha='right')

conds = [x for x in ["0%", "25%", "50%", "75%", "100%"] for _ in range(len(subjects))]
analyse_pd = pd.DataFrame({'Electric shock intensity': conds, 'VAS pain ratings': pain_rating_by_conds})

print(analyse_pd.to_string())

ort = 'v'

dx = 'Electric shock intensity'
dy = 'VAS pain ratings'

pal = 'flare'

ax=pt.half_violinplot( x = dx, y = dy, data = analyse_pd, palette=pal,
      bw = 0.5, cut = 0.,scale = "area", width = 20, inner = None, 
      orient = ort, alpha = 0.8, linewidth=0)

ax=sns.stripplot( x = dx, y = dy, data = analyse_pd, palette=pal,
     edgecolor = "white",size = 8, jitter = 1, zorder = 0, 
     orient = ort)

ax=sns.boxplot( x = dx, y = dy, data = analyse_pd, palette=pal,
      width = .25, zorder = 10, showcaps = True,
      boxprops = {'facecolor': [0, 0, 0, 0], "zorder": 10}, showfliers=True,
      whiskerprops = {'linewidth':2, "zorder":10}, 
      saturation = 1, orient = ort)

plt.gca().tick_params(axis='both', labelsize=24)
sns.despine(offset=0, trim=True)

plt.xlabel('Electric shock intensity', fontsize=36)
plt.ylabel('VAS pain ratings', fontsize=36)

ax = fig.add_subplot(1, 3, 2)
ax.text(-0.1, 1.15, 'd', transform=ax.transAxes,
    fontsize=42, fontweight='bold', va='top', ha='right')

def average_func(yss):
    np_yss = [np.array(ys) for ys in yss]
    np_yss_mat = np.stack(np_yss)
    return np.mean(np_yss_mat, axis=0)

def func_compute(params, xs):
    target_func = get_pain_func(*params)
    return [target_func(x) for x in xs]

xs = np.arange(-0.2, 1.2, 0.001)
yss = [func_compute(v, xs) for k, v in results.items()]

cmap = mpl.cm.get_cmap('Spectral')
norm = mpl.colors.Normalize(vmin=-1, vmax=0)
for ys in yss:
    plt.plot(xs, ys, alpha=1, color=cmap(norm(ys[-1])))

levels = [0, 0.25, 0.5, 0.75, 1]
plt.xticks(levels, labels=["0%", "25%", "50%", "75%", "100%"])
for level in levels:
    plt.axvline(level, linestyle='dashed')
avg_ys = average_func(yss)
plt.plot(xs, avg_ys, c='black', linewidth=3)
plt.gca().invert_yaxis()

plt.xlabel('Electric shock intensity', fontsize=36)
plt.ylabel('Phasic pain utility function $C_p\ u$', fontsize=36)
plt.gca().tick_params(axis='both', labelsize=24)

sns.despine(offset=0, trim=True)
cbar_ax = fig.add_axes([0.40, 0.85, 0.1, 0.03])    
mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
cbar_ax.set_title('At 100% shock intensity', fontsize=24)
cbar_ax.invert_xaxis()

ax2 = fig.add_subplot(1, 3, 3, sharey=ax)
ax2.text(-0.1, 1.15, 'e', transform=ax2.transAxes,
    fontsize=42, fontweight='bold', va='top', ha='right')

dx = pain_rating_by_conds
dy = [get_pain_func(*v)(i * 0.25) for i in range(5) for k, v in results.items()]

sns.regplot(x=dx, y=dy, ax=ax2, scatter_kws={'s': 48})
sns.despine(offset=0, trim=True, left=True, ax=ax2)
plt.gca().tick_params(axis='x', labelsize=24)
ax2.tick_params(labelleft=False, left=False)
ax2.set_xlim((-1, 11))
plt.xlabel('VAS pain ratings', fontsize=36)

transFigure = fig.transFigure.inverted()
for i, pain_func_v in enumerate(dy):
    coord1 = transFigure.transform(ax.transData.transform([int(i / len(subjects)) * 0.25, pain_func_v]))
    coord2 = transFigure.transform(ax2.transData.transform([dx[i], pain_func_v]))
    line = mpl.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure, alpha=0.1, c='black')
    fig.lines.append(line)

plt.savefig('figures/PUB/Phasic_Pain_Utility_Function_To_Ratings.svg', bbox_inches='tight')

import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

base = importr("base")
rstats = importr("stats")

df = pd.DataFrame({"Ratings": dx, "PainFunc": dy})
print(df)
with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(df)

fml = ro.Formula("PainFunc ~ Ratings")
aov = rstats.aov(fml, data=r_from_pd_df)
print(base.summary(aov))
lm = rstats.lm(fml, data=r_from_pd_df)
print(base.summary(lm))
