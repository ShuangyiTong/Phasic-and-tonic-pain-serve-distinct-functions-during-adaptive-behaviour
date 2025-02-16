# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import core.utils
import sys
import numpy as np
import pandas as pd
import math

core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_NAME, UNITY_DEVICE_ID, ARDUINO_DEVICE_NAME
from core.experiment_data import set_expt

Expt_Name = sys.argv[1]
set_expt(Expt_Name)

if Expt_Name == 'Expt4':
    start_trial_for_analysis = 6
    end_trial_for_analysis = 24
else:
    start_trial_for_analysis = -10
    end_trial_for_analysis = None

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series

from core.individual_subject import get_end_of_trial_pain_ratings
from core.individual_subject import get_fruit_picking_moving_distance
from core.individual_subject import apply_corrections_natural_number_indexing
from core.individual_subject import get_fruit_position_map
from core.individual_subject import fruit_to_prev_gaze_head_distance

from core.plot import all_curves, plot_all_subjects_pain_value_curve

from core.utils import save_cache, load_cache

import matplotlib.pyplot as plt

dicts = None #load_cache('distance_dicts')
if dicts is None:
    exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB11"], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME])
else:
    exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB11"], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME])


subjects = list(exp_data.keys())

all_pain_ratings = get_multiple_series(exp_data, lambda individual_data: apply_corrections_natural_number_indexing(
                                                    individual_data, get_end_of_trial_pain_ratings(individual_data), 
                                                    'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis], subjects)

if dicts is None:
    painful_distance_bias = get_multiple_series(exp_data, lambda individual_data: [np.mean(x) - np.mean(y) 
                                                                                        if x==x and y==y else math.nan # check if it is already math.nan (no pick ups)
                                                                                    for x, y in zip(get_fruit_picking_moving_distance(individual_data, 
        get_fruit_position_map(individual_data), lambda x: not x.endswith('G'), fruit_to_prev_gaze_head_distance, use_average=False), get_fruit_picking_moving_distance(individual_data, 
        get_fruit_position_map(individual_data), lambda x: x.endswith('G'), fruit_to_prev_gaze_head_distance, use_average=False))][start_trial_for_analysis:end_trial_for_analysis], subjects)

    curves = all_curves(all_pain_ratings, painful_distance_bias)

    dicts = [[{ 'bias' : bias, 'participant' : participant_index, 'rating' : pain_rating, } for pain_rating, bias in zip(pain_ratings, distance_bias)] 
        for participant_index, pain_ratings, distance_bias in zip(range(len(all_pain_ratings)), all_pain_ratings, painful_distance_bias)]

    print(painful_distance_bias)
    save_cache(dicts, 'distance_dicts')

analyse_pd = pd.DataFrame()
dropped_because_of_nan = 0
dropped_because_of_high = 0
remaining = 0
for participant in dicts:
    for block in participant:
        if block['bias'] != block['bias']:
            dropped_because_of_nan += 1
        else:
            analyse_pd = analyse_pd.append(block, ignore_index=True)
            remaining += 1

print("Dropped for NaN", dropped_because_of_nan)
print("Remaining", remaining)
assert(dropped_because_of_high + dropped_because_of_nan + remaining == (abs(start_trial_for_analysis) if end_trial_for_analysis == None else end_trial_for_analysis - start_trial_for_analysis) * len(subjects))

import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

base = importr("base")
lme4 = importr('lme4')
lmer_test = importr('lmerTest')
stats = importr('stats')
broom = importr('broom.mixed')
generics = importr('generics')
import rpy2.robjects.lib.ggplot2 as ggplot2

with localconverter(ro.default_converter + pandas2ri.converter):
  r_from_pd_df = ro.conversion.py2rpy(analyse_pd)


m = lmer_test.lmer('bias ~ rating + (1 + rating | participant)', r_from_pd_df)
print(base.summary(m))
print(stats.confint(m))
print(generics.tidy(m))
print(base.warnings())

pp = ggplot2.ggplot(r_from_pd_df) + ggplot2.aes_string(x='rating', y='bias') + ggplot2.geom_point() + ggplot2.geom_smooth(method='lm')

pp.save('figures/PUB/Ratings_vs_distance_bias_simple_linear_regression.png')

import matplotlib.colors as cl
import matplotlib.cm as cm
import seaborn as sns
import matplotlib as mpl
sns.set_theme()
sns.set_style("ticks")
cmap = mpl.cm.get_cmap('Spectral')


from core.plot import all_curves

curves = all_curves(all_pain_ratings, painful_distance_bias)

def plot_all_subjects_pain_value_curve(curves, intercept, slope):
    fig = plt.figure(figsize=(9, 6))
    norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.2)
    for curve in curves:
        new_curve_r = [r for r, c in zip(curve[0], curve[1]) if c == c]
        new_curve_c = [c for c in curve[1] if c == c]
        print(np.mean(new_curve_c))
        plt.scatter(new_curve_r, new_curve_c, color=cmap(norm(new_curve_c[0])), alpha=0.7)
        plt.plot(new_curve_r, new_curve_c, color=cmap(norm(new_curve_c[0])), alpha=0.4)

    plt.ylabel("Choice Distance Bias (m)", fontsize=18)
    plt.xlabel("Visual Analogue Scale Pain Ratings", fontsize=18)
    plt.plot([0, 10], [intercept, slope * 10 + intercept], color='black', linestyle='dashed', label='Linear mixed model fitted line\n (p=0.0734, t=1.866, df=26.012)')

    sns.despine(offset=0, trim=True)
    cbar_ax = fig.add_axes([0.15, 0.77, 0.3, 0.03])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cbar_ax.set_title('At VAS=0', fontsize=18)
    # plt.tight_layout()
    plt.savefig('figures/PUB/Ratings_vs_distance_bias_lmer' + Expt_Name + '.svg')

# lmer result with bias ~ rating + (1 + rating | participant), check this agrees with the output from R
intercept = 0.04571
slope = 0.01133

plot_all_subjects_pain_value_curve(curves, intercept, slope)
