# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION
from core.experiment_data import set_expt

EXPT_NAME = 'Expt2'
set_expt(EXPT_NAME)

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_lazy

from core.individual_subject import get_end_of_trial_pain_ratings
from core.individual_subject import apply_corrections_natural_number_indexing
from core.individual_subject import get_all_collected_fruits
from core.individual_subject import get_series_from_control

from core.utils import save_cache, load_cache

import matplotlib.pyplot as plt

pain_cond_idx = [ "NoPain", "MidLowPain", "MidMidPain", "MidHighPain", "MaxPain" ]
if EXPT_NAME == 'Expt3':
    pain_cond_idx = [ "NoPain", "HighFinger", "LowFinger", "HighBack", "LowBack" ]
elif EXPT_NAME == 'Expt4':
    pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB11"], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION], lazy_closure=True)

subjects = list(exp_data.keys())

start_trial_for_analysis = -10
end_trial_for_analysis = None

all_pain_ratings = get_multiple_series_lazy(exp_data, lambda individual_data: apply_corrections_natural_number_indexing(
                                                    individual_data, get_end_of_trial_pain_ratings(individual_data), 
                                                    'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis], subjects)

all_collection_bias = get_multiple_series_lazy(exp_data, lambda individual_data: [sum(map(lambda x : x.endswith('G'), fruits)) / len(fruits) 
                                                                                for fruits in get_all_collected_fruits(individual_data)][start_trial_for_analysis:end_trial_for_analysis], subjects)
                            
all_pain_conds = get_multiple_series_lazy(exp_data, lambda individual_data: list(map(
                    lambda msg: msg.split('-')[-1], 
                    get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], subjects)

import scipy.stats as stats 
import numpy as np

dicts = [[{ 'choice_prob' : bias, 'participant' : participant_index, 'rating' : pain_rating, } for pain_rating, bias in zip(pain_ratings, bias)] 
    for participant_index, pain_ratings, bias in zip(subjects, all_pain_ratings, all_collection_bias)]

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
stats = importr('stats')
broom = importr('broom.mixed')
generics = importr('generics')
import rpy2.robjects.lib.ggplot2 as ggplot2

with localconverter(ro.default_converter + pandas2ri.converter):
  r_from_pd_df = ro.conversion.py2rpy(analyse_pd)

m = lmer_test.lmer('choice_prob ~ rating + (1 + rating | participant)', r_from_pd_df)

print(base.summary(m))
print(stats.confint(m))
print(generics.tidy(m))
print(stats.anova(m))

pp = ggplot2.ggplot(r_from_pd_df) + ggplot2.aes_string(x='rating', y='choice_prob') + ggplot2.geom_point() + ggplot2.geom_smooth(method='lm')

pp.save('figures/PUB/Ratings_vs_pain_choice_prob_simple_linear_regression.png')

import matplotlib.colors as cl
import matplotlib.cm as cm
import seaborn as sns
import matplotlib as mpl
sns.set_theme()
sns.set_style("ticks")
cmap = mpl.cm.get_cmap('Spectral')


from core.plot import all_curves

curves = all_curves(all_pain_ratings, all_collection_bias)

def plot_all_subjects_pain_value_curve(curves, intercept, slope):
    curve_val = [[] for i in range(11)]
    fig = plt.figure(figsize=(9, 6))
    norm = mpl.colors.Normalize(vmin=0.25, vmax=0.65)
    for curve in curves:
        for r, c in zip(curve[0], curve[1]):
            if c == c:
                curve_val[r].append(c)
        
        new_curve_r = [r for r, c in zip(curve[0], curve[1]) if c == c]
        new_curve_c = [c for c in curve[1] if c == c]
        print(np.mean(new_curve_c))
        plt.scatter(new_curve_r, new_curve_c, color=cmap(norm(new_curve_c[0])), alpha=0.7)
        plt.plot(new_curve_r, new_curve_c, color=cmap(norm(new_curve_c[0])), alpha=0.4)

    plt.ylabel("Aversive Choice Probability", fontsize=18)
    plt.xlabel("Visual Analogue Scale Pain Ratings", fontsize=18)
    plt.plot([0, 10], [intercept, slope * 10 + intercept], color='black', linestyle='dashed', label='Linear mixed model fitted line\n (p=0.0734, t=1.866, df=26.012)')

    sns.despine(offset=0, trim=True)
    if EXPT_NAME == 'Expt4':
        cbar_ax = fig.add_axes([0.68, 0.92, 0.3, 0.03])
    else:
        cbar_ax = fig.add_axes([0.15, 0.17, 0.3, 0.03])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cbar_ax.set_title('At VAS=0', fontsize=18)
    # plt.tight_layout()
    plt.savefig('figures/PUB/Ratings_vs_pain_choice_prob_lmer' + EXPT_NAME + '.svg')
    plt.savefig('figures/PUB/Ratings_vs_pain_choice_prob_lmer' + EXPT_NAME + '.png', dpi=1200)

intercept = 0.451944
slope = -0.026305

plot_all_subjects_pain_value_curve(curves, intercept, slope)
