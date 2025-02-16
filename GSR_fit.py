# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import sys
import os
import dill

from core.utils import nan_square_sum

import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.plot import scattered_boxplot

Expt_name = sys.argv[1]

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_NAME, NI_DEVICE_ID, ARDUINO_DEVICE_NAME
from core.utils import save_cache, load_cache

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series

from core.individual_subject import get_series_from_control
from core.individual_subject import get_trial_start_timestamps
from core.individual_subject import get_gsr_segments_with_crfs
from core.individual_subject import apply_corrections_natural_number_indexing
from core.individual_subject import get_end_of_trial_pain_ratings

from core.experiment_data import set_expt

set_expt(Expt_name)

segments = 1
CACHE_FILE_NAME = 'gsr_fit_data'
stimulation_result_file_name = 'temp/' + Expt_name + '_fit_fin.dill'
R_FITTING_FOLDER = 'temp/' + CACHE_FILE_NAME

all_gsr_fit_data = None #load_cache(CACHE_FILE_NAME)
if all_gsr_fit_data == None:
    exclude_data = [NI_DEVICE_NAME]
else:
    exclude_data = [UNITY_DEVICE_NAME, NI_DEVICE_NAME, ARDUINO_DEVICE_NAME]
exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB11"], exclude_device_data=exclude_data)

subjects = list(exp_data.keys())
if Expt_name == 'Expt4':
    start_trial_for_analysis = 6
    end_trial_for_analysis = 24
else:
    start_trial_for_analysis = -10
    end_trial_for_analysis = None
    
pain_cond_idx = [ "NoPain", "MidLowPain", "MidMidPain", "MidHighPain", "MaxPain" ]

if all_gsr_fit_data is None:
    all_gsr_fit_data = get_multiple_series(exp_data, lambda individual_data:
                                                   get_gsr_segments_with_crfs(individual_data, 
                                                                    get_trial_start_timestamps(individual_data)
                                                                    [start_trial_for_analysis:end_trial_for_analysis],
                                                                    list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))
                                                                    [start_trial_for_analysis:end_trial_for_analysis],
                                                                    segment_per_block=segments,
                                                                    baseline_window='average', min_ms_passed_per_uS_change=None), subjects)

    def segmented_plot(series, c='red', label='None'):
        SEGMENT_LENGTH = int((60 * 10) / segments) # 60s * 10Hz resample freq
        i = 0
        while i < len(series):
            plt.plot([x + i for x in range(SEGMENT_LENGTH)], series[i: i + SEGMENT_LENGTH], color=c, label=label)
            i += SEGMENT_LENGTH

    def plot_gsr_fitting_by_block(analyse_pd, fname, coef=None):
        segmented_plot(analyse_pd['GSR'], c='black', label='GSR')
        if coef:
            if len(coef) > 3:
                segmented_plot(coef[0] * analyse_pd['Seen_Green'] + coef[1] * analyse_pd['Pick_Green']
                            + coef[2] * analyse_pd['Seen_Yellow'] + coef[3] * analyse_pd['Pick_Yellow'] + coef[4], c='blue', label='Fitted')
            else:
                segmented_plot(coef[0] * analyse_pd['Seen_Green'] + coef[1] * analyse_pd['Pick_Green'] + coef[2], c='blue', label='Fitted')

        else:
            segmented_plot(analyse_pd['Seen_Green'], c='orange', label='Seen')
            segmented_plot(analyse_pd['Pick_Green'], c='red', label='Pick')
        plt.legend()
        plt.savefig(fname)
        plt.close()

    def send_r_for_glm_fitting(gsr_fit_data, subject):
        seen_blockwise_data = []
        pick_blockwise_data = []
        seen_blockwise_std = []
        pick_blockwise_std = []
        seen_blockwise_p = []
        pick_blockwise_p = []

        os.makedirs(R_FITTING_FOLDER, exist_ok=True)

        def append_nan():
            seen_blockwise_data.append(math.nan)
            pick_blockwise_data.append(math.nan)
            seen_blockwise_std.append(math.nan)
            pick_blockwise_std.append(math.nan)
            seen_blockwise_p.append(math.nan)
            pick_blockwise_p.append(math.nan)

        for block in range((end_trial_for_analysis if end_trial_for_analysis else 0) - start_trial_for_analysis):
            data_dict = { "GSR": gsr_fit_data[0][block], "Seen_Green": gsr_fit_data[1][block], "Pick_Green": gsr_fit_data[2][block],
                              "Seen_Yellow": gsr_fit_data[3][block], "Pick_Yellow": gsr_fit_data[4][block] }

            if len(gsr_fit_data[0][block]) == 0:
                append_nan()
                continue
            
            R_INPUT = R_FITTING_FOLDER + '/to_r_dataframe_' + subject + '_' + str(block)
            analyse_pd = pd.DataFrame(data_dict)
            analyse_pd.to_csv(R_INPUT)

            plot_gsr_fitting_by_block(analyse_pd, R_FITTING_FOLDER + '/prefit_plot_' + subject + '_' + str(block) + '.png')

            R_OUTPUT = R_FITTING_FOLDER + '/to_py_summary_' + subject + '_' + str(block)
            # Must call R externally and parse  as Rpy2 is unable to handle NA output
            os.system("rscript gsr_glm_fit.r " + R_INPUT + ' ' + R_OUTPUT)

            if os.path.exists(R_OUTPUT):
                with open(R_OUTPUT, 'r') as f:
                    summary = f.read()
            else:
                append_nan()
            
            lines = summary.split('\n')
            seen_coef = None
            pick_coef = None
            seen_coef_yellow = None
            pick_coef_yellow = None
            for line in lines:
                if line.startswith('Seen_Green') or line.startswith('Pick_Green') or line.startswith('Seen_Yellow') or line.startswith('Pick_Yellow') or line.startswith('(Intercept)'):
                    vals = line.split()
                    print(vals)
                    if vals[1] == 'NA':
                        coef = math.nan
                        std = math.nan
                        p = math.nan
                    else:
                        coef, std, t, p = float(vals[1]), float(vals[2]), float(vals[3]), vals[4]
                        if p.startswith('<'):
                            p = 0
                        else:
                            p = float(p)

                    if line.startswith('Seen_Green'):
                        seen_coef = coef
                        seen_std = std
                        seen_p = p
                    elif line.startswith('Pick_Green'):
                        pick_coef = coef
                        pick_std = std
                        pick_p = p
                    elif line.startswith('Seen_Yellow'):
                        seen_coef_yellow = coef
                        seen_p_yellow = p
                    elif line.startswith('Pick_Yellow'):
                        pick_coef_yellow = coef
                        pick_p_yellow = p
                    else:
                        intercept = coef

            # will throw error for uncaptured coefs which is None
            # plot_gsr_fitting_by_block(analyse_pd, R_FITTING_FOLDER + '/postfit_plot_' + subject + '_' + str(block) + '.png', [seen_coef, pick_coef, seen_coef_yellow, pick_coef_yellow, intercept])
            plot_gsr_fitting_by_block(analyse_pd, R_FITTING_FOLDER + '/postfit_plot_' + subject + '_' + str(block) + '.png', [seen_coef, pick_coef, intercept])
            seen_blockwise_data.append(seen_coef)
            pick_blockwise_data.append(pick_coef)
            seen_blockwise_std.append(seen_std)
            pick_blockwise_std.append(pick_std)
            seen_blockwise_p.append(seen_p)
            pick_blockwise_p.append(pick_p)
            print(seen_blockwise_data, pick_blockwise_data, seen_blockwise_std, pick_blockwise_std, seen_blockwise_p, pick_blockwise_p)
            assert(len(seen_blockwise_data) == len(pick_blockwise_data) == len(seen_blockwise_std) == len(pick_blockwise_std) == len(seen_blockwise_p) == len(pick_blockwise_p))

        return (seen_blockwise_data, pick_blockwise_data, seen_blockwise_std, pick_blockwise_std, seen_blockwise_p, pick_blockwise_p)

    all_seen_coefs = []
    all_pick_coefs = []
    all_seen_stds = []
    all_pick_stds = []
    all_seen_p = []
    all_pick_p = []

    for i, subject in enumerate(subjects):
        res = send_r_for_glm_fitting(all_gsr_fit_data[i], subject)

        all_seen_coefs.append(res[0])
        all_pick_coefs.append(res[1])
        all_seen_stds.append(res[2])
        all_pick_stds.append(res[3])
        all_seen_p.append(res[4])
        all_pick_p.append(res[5])
    
    save_cache((all_seen_coefs, all_pick_coefs, all_seen_stds, all_pick_stds, all_seen_p, all_pick_p, subjects), CACHE_FILE_NAME)
else:
    all_seen_coefs = all_gsr_fit_data[0]
    all_pick_coefs = all_gsr_fit_data[1]
    all_seen_stds = all_gsr_fit_data[2]
    all_pick_stds = all_gsr_fit_data[3]
    all_seen_p = all_gsr_fit_data[4]
    all_pick_p = all_gsr_fit_data[5]
    
    # ensures the order is the same as current subjects to match data fitting results dict
    assert(subjects == all_gsr_fit_data[6])

all_pain_conds = get_multiple_series(exp_data, lambda individual_data: list(map(
    lambda msg: msg.split('-')[-1], 
    get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], subjects)

print(len(all_seen_coefs), len(all_pick_coefs), len(all_seen_stds), len(all_pick_stds))

seen_coefs_by_conds = [[seen_coef for seen_coefs, pain_conds in zip(all_seen_coefs, all_pain_conds) for seen_coef, pain_cond in zip(seen_coefs, pain_conds) if pain_cond == target_cond] 
                                for target_cond in pain_cond_idx]

pick_coefs_by_conds = [[pick_coef for pick_coefs, pain_conds in zip(all_pick_coefs, all_pain_conds) for pick_coef, pain_cond in zip(pick_coefs, pain_conds) if pain_cond == target_cond] 
                                for target_cond in pain_cond_idx]

seen_stds_by_conds = [[seen_coef for seen_coefs, pain_conds in zip(all_seen_stds, all_pain_conds) for seen_coef, pain_cond in zip(seen_coefs, pain_conds) if pain_cond == target_cond]
                                for target_cond in pain_cond_idx]

pick_stds_by_conds = [[pick_coef for pick_coefs, pain_conds in zip(all_pick_stds, all_pain_conds) for pick_coef, pain_cond in zip(pick_coefs, pain_conds) if pain_cond == target_cond]
                                for target_cond in pain_cond_idx]

seen_p_by_conds = [[seen_coef for seen_coefs, pain_conds in zip(all_seen_p, all_pain_conds) for seen_coef, pain_cond in zip(seen_coefs, pain_conds) if pain_cond == target_cond]
                                for target_cond in pain_cond_idx]

pick_p_by_conds = [[pick_coef for pick_coefs, pain_conds in zip(all_pick_p, all_pain_conds) for pick_coef, pain_cond in zip(pick_coefs, pain_conds) if pain_cond == target_cond]
                                for target_cond in pain_cond_idx]

print(len(seen_coefs_by_conds[0]), len(seen_coefs_by_conds[1]), len(seen_coefs_by_conds[2]), len(seen_coefs_by_conds[3]), len(seen_coefs_by_conds[4]))
print(len(pick_coefs_by_conds[0]), len(pick_coefs_by_conds[1]), len(pick_coefs_by_conds[2]), len(pick_coefs_by_conds[3]), len(pick_coefs_by_conds[4]))
print(len(seen_stds_by_conds[0]), len(seen_stds_by_conds[1]), len(seen_stds_by_conds[2]), len(seen_stds_by_conds[3]), len(seen_stds_by_conds[4]))
print(len(pick_stds_by_conds[0]), len(pick_stds_by_conds[1]), len(pick_stds_by_conds[2]), len(pick_stds_by_conds[3]), len(pick_stds_by_conds[4]))

# scattered_boxplot(plt.gca(), seen_coefs_by_conds)
# plt.show()

# scattered_boxplot(plt.gca(), pick_coefs_by_conds)
# plt.show()

all_stimulation_results = None
with open(stimulation_result_file_name, 'rb') as f:
    all_stimulation_results = dill.load(f)

def get_pain_scale(positivity, horizontal, vertical):
    return positivity * (1 - math.sqrt(nan_square_sum([horizontal, vertical])))

def get_pain_func(param1, param2, positivity, horizontal, vertical):
    scale = get_pain_scale(positivity, horizontal, vertical)
    return lambda x: scale / (1 + math.exp(-param1 * (x - param2)))

results = {}
detailed_results = {}
results_index = 1

for subject_name, stimulation_results in zip(all_stimulation_results[0], all_stimulation_results[1]):
    results[subject_name] = list(stimulation_results[results_index][0])
    print(subject_name)

results_subject_order = [results[subject] for subject in subjects]

final_pd = pd.DataFrame()

def func_compute(params, x):
    target_func = get_pain_func(*params)
    return target_func(x)

all_ratings = get_multiple_series(exp_data, lambda individual_data: apply_corrections_natural_number_indexing(individual_data, get_end_of_trial_pain_ratings(individual_data), 'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis], subjects)
pain_rating_by_conds = [[rating for ratings, pain_conds in zip(all_ratings, all_pain_conds) for rating, pain_cond in zip(ratings, pain_conds) if pain_cond == target_cond]
                                for target_cond in pain_cond_idx]

for i, (seen_coefs_by_cond, pick_coefs_by_cond, seen_stds_by_cond, pick_stds_by_cond, seen_p_by_cond, pick_p_by_cond, ratings_by_cond) in enumerate(zip(seen_coefs_by_conds,
                                                                                  pick_coefs_by_conds,
                                                                                  seen_stds_by_conds,
                                                                                  pick_stds_by_conds,
                                                                                  seen_p_by_conds,
                                                                                  pick_p_by_conds,
                                                                                  pain_rating_by_conds)):
    for subject_id, (seen_coef, pick_coef, seen_std, pick_std, seen_p, pick_p, rating) in enumerate(zip(seen_coefs_by_cond, 
                                                                    pick_coefs_by_cond,
                                                                    seen_stds_by_cond,
                                                                    pick_stds_by_cond,
                                                                    seen_p_by_cond,
                                                                    pick_p_by_cond,
                                                                    ratings_by_cond)):
        print(int(subject_id / 2)) # always two blocks per condition per subject, order follows subjects
        final_pd = final_pd.append({'Subject': str(int(subject_id / 2)),
                                    'block_id': str(i),
                                    'seen_coef': seen_coef,
                                    'pick_coef': pick_coef,
                                    'seen_var': seen_std * seen_std,
                                    'pick_var': pick_std * pick_std,
                                    'seen_p': seen_p,
                                    'pick_p': pick_p,
                                    'rating': rating,
                                    'PainFunc': func_compute(results_subject_order[int(subject_id / 2)], i * 0.25),
                                      }, ignore_index=True)

print(final_pd.to_string())

analyse_pd_seen = final_pd[(final_pd['seen_p'] < 0.05)]
analyse_pd_seen = analyse_pd_seen[np.abs(stats.zscore(analyse_pd_seen['seen_coef'], nan_policy='omit')) < 3]
analyse_pd_pick = final_pd[(final_pd['pick_p'] < 0.05)]
analyse_pd_pick = analyse_pd_pick[np.abs(stats.zscore(analyse_pd_pick['pick_coef'], nan_policy='omit')) < 3]

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
rstatix = importr('rstatix')

with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(analyse_pd_seen)

effective_counts = len(analyse_pd_seen[analyse_pd_seen['seen_coef'] == analyse_pd_seen['seen_coef']])
print("Effective counts:", effective_counts)
print('PainFunc Mean: ', np.mean(analyse_pd_seen['PainFunc']), ' SD:', np.std(analyse_pd_seen['PainFunc']))
print('rating Mean: ', np.mean(analyse_pd_seen['rating']), ' SD:', np.std(analyse_pd_seen['rating']))
print('seen_coef Mean: ', np.mean(analyse_pd_seen['seen_coef']), ' SD:', np.std(analyse_pd_seen['seen_coef']))

m = rstats.lm('seen_coef ~ PainFunc', r_from_pd_df)
print(base.summary(m))
print(rstats.anova(m))
print("CORRELATION", stats.pearsonr(analyse_pd_seen['PainFunc'], analyse_pd_seen['seen_coef']))

fml = ro.Formula('seen_coef ~ PainFunc')
aov = rstats.aov(fml, data=r_from_pd_df)
print(base.summary(aov))

pp = ggplot2.ggplot(r_from_pd_df) + ggplot2.aes_string(x='seen_coef', y='PainFunc') + ggplot2.geom_point() + ggplot2.geom_smooth(method='lm', mapping=ggplot2.aes())

pp.save('figures/PUB/PainFunc_GSR_seen_coef.png')

m = rstats.lm('seen_coef ~ rating', r_from_pd_df)
print(base.summary(m))
print(rstats.anova(m))
print("CORRELATION", stats.pearsonr(analyse_pd_seen['rating'], analyse_pd_seen['seen_coef']))

fml = ro.Formula('seen_coef ~ rating')
aov = rstats.aov(fml, data=r_from_pd_df)
print(base.summary(aov))

pp = ggplot2.ggplot(r_from_pd_df) + ggplot2.aes_string(x='seen_coef', y='rating') + ggplot2.geom_point() + ggplot2.geom_smooth(method='lm', mapping=ggplot2.aes())

pp.save('figures/PUB/rating_GSR_seen_coef.png')

with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(analyse_pd_pick)

print("Effective counts:", len(analyse_pd_pick[analyse_pd_pick['pick_coef'] == analyse_pd_pick['pick_coef']]))
print('PainFunc Mean: ', np.mean(analyse_pd_pick['PainFunc']), ' SD:', np.std(analyse_pd_pick['PainFunc']))
print('rating Mean: ', np.mean(analyse_pd_pick['rating']), ' SD:', np.std(analyse_pd_pick['rating']))
print('pick_coef Mean: ', np.mean(analyse_pd_pick['pick_coef']), ' SD:', np.std(analyse_pd_pick['pick_coef']))

m = rstats.lm('pick_coef ~ PainFunc', r_from_pd_df)
print(base.summary(m))
print(rstats.anova(m))
print("CORRELATION", stats.pearsonr(analyse_pd_pick['PainFunc'], analyse_pd_pick['pick_coef']))

fml = ro.Formula('pick_coef ~ PainFunc')
aov = rstats.aov(fml, data=r_from_pd_df)
print(base.summary(aov))

pp = ggplot2.ggplot(r_from_pd_df) + ggplot2.aes_string(x='pick_coef', y='PainFunc') + ggplot2.geom_point() + ggplot2.geom_smooth(method='lm', mapping=ggplot2.aes())

pp.save('figures/PUB/PainFunc_GSR_pick_coef.png')

m = rstats.lm('pick_coef ~ rating', r_from_pd_df)
print(base.summary(m))
print(rstats.anova(m))
print("CORRELATION", stats.pearsonr(analyse_pd_pick['rating'], analyse_pd_pick['pick_coef']))

fml = ro.Formula('pick_coef ~ rating')
aov = rstats.aov(fml, data=r_from_pd_df)
print(base.summary(aov))

pp = ggplot2.ggplot(r_from_pd_df) + ggplot2.aes_string(x='pick_coef', y='rating') + ggplot2.geom_point() + ggplot2.geom_smooth(method='lm', mapping=ggplot2.aes())

pp.save('figures/PUB/rating_GSR_pick_coef.png')

import seaborn as sns

fig = plt.figure(figsize=(22, 10))
plt.subplots_adjust(wspace=0.5)
ax = fig.add_subplot(1, 2, 1)
ax.text(-0.1, 1.16, 'a', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')
sns.regplot(x='seen_coef', y='PainFunc', ax=ax, data=analyse_pd_seen, scatter_kws={'s': 48})
#ax.set_ylim(-1.1, 0.2)
ax.set_xlim(-0.19, 0.15)
ax.invert_yaxis()
ax.set_xlabel('Fixation evoked SCR fitted coefficients', fontsize=24)
ax.set_ylabel('Phasic pain utility function value (Decision values)', fontsize=24)
ax.tick_params(axis='both', labelsize=24)
ax2 = ax.twinx()
sns.regplot(x='seen_coef', y='rating', ax=ax2, data=analyse_pd_seen, color='orange', scatter_kws={'s': 48})
ax2.set_ylim(-1, 11)
ax2.set_ylabel('Subjective VAS pain ratings', fontsize=24)
ax2.tick_params(axis='both', labelsize=24)

sns.despine(offset=0, trim=True, right=False, ax=ax)
sns.despine(offset=0, trim=True, right=False, ax=ax2)

ax = fig.add_subplot(1, 2, 2)
ax.text(-0.1, 1.16, 'b', transform=ax.transAxes,
    fontsize=32, fontweight='bold', va='top', ha='right')
sns.regplot(x='pick_coef', y='PainFunc', ax=ax, data=analyse_pd_pick, scatter_kws={'s': 48})
ax.set_xlabel('Shock evoked SCR fitted coefficients', fontsize=24)
ax.set_xlim(-0.34, 0.6)
ax.set_ylabel('Phasic pain utility function value (Decision values)', fontsize=24)
#ax.set_ylim(-1.1, 0.2)
ax.invert_yaxis()
ax.tick_params(axis='both', labelsize=24)
ax2 = ax.twinx()
sns.regplot(x='pick_coef', y='rating', ax=ax2, data=analyse_pd_pick, color='orange', scatter_kws={'s': 48})
ax2.set_ylabel('Subjective VAS pain ratings', fontsize=24)
ax2.set_ylim(-1, 11)
ax2.tick_params(axis='both', labelsize=24)

import matplotlib.patches as mpatches
pain_patch = mpatches.Patch(color='C0', alpha=0.7, label='Decision values')
nopain_patch = mpatches.Patch(color='orange', alpha=0.7, label='Subjective VAS pain ratings')
plt.gcf().legend(handles=[pain_patch, nopain_patch], fontsize=24, loc='upper right')

sns.despine(offset=0, trim=True, right=False, ax=ax)
sns.despine(offset=0, trim=True, right=False, ax=ax2)
ax2.tick_params(axis='y', right=True) # stupid despine https://stackoverflow.com/questions/62042866/how-to-stop-seaborn-despine-removing-y-tick-marks-on-second-axis
ax2.set_yticks([0, 2, 4, 6, 8, 10])

plt.savefig("figures/PUB/seen_coef.svg", bbox_inches='tight')