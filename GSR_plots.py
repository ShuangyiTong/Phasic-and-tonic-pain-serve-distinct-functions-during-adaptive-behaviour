# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import sys

Expt_name = sys.argv[1]

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_NAME, NI_DEVICE_ID, ARDUINO_DEVICE_NAME
from core.utils import save_cache, load_cache

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_lazy

from core.individual_subject import get_series_from_control
from core.individual_subject import get_labelled_pick_up_action_timestamp
from core.individual_subject import get_trial_start_timestamps
from core.individual_subject import get_raw_arduino_data
from core.individual_subject import grove_gsr_12_bit_adc_processing

from core.experiment_data import set_expt

set_expt(Expt_name)

CACHE_FILE_NAME = 'gsr_with_labels'
DATA_FIELD = 'skin_conductance'
TIME_TO_PREV = 300
INTERVAL_WIDTH = 100
TOTAL_TIME = 4700
REQUIRED_TIME = TOTAL_TIME * 0.8
INTERVALS = int(TOTAL_TIME / INTERVAL_WIDTH)

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB11"], exclude_device_data=[NI_DEVICE_NAME], lazy_closure=True)

subjects = list(exp_data.keys())
start_trial_for_analysis = -10
end_trial_for_analysis = None

pain_cond_idx = [ "NoPain", "MidLowPain", "MidMidPain", "MidHighPain", "MaxPain" ]

all_gsr_with_labels = None #load_cache(CACHE_FILE_NAME)
if all_gsr_with_labels is None:
    all_gsr_with_labels_seen = get_multiple_series_lazy(exp_data, lambda individual_data:
        (lambda labelled_ts: [(label, is_pain, grove_gsr_12_bit_adc_processing(get_raw_arduino_data(individual_data, DATA_FIELD, ts - TIME_TO_PREV, TOTAL_TIME, global_sync_ts),
                                                                                 baseline_window=2,
                                                                                 required_length=REQUIRED_TIME,
                                                                                 resample_length=(ts - TIME_TO_PREV, INTERVAL_WIDTH, INTERVALS)))
            for label, is_pain, global_sync_ts, ts in labelled_ts])
            (get_labelled_pick_up_action_timestamp(individual_data, 
                list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], 
                get_trial_start_timestamps(individual_data)[start_trial_for_analysis:end_trial_for_analysis], 'first_seen', True)), subjects)    
    all_gsr_with_labels_pick = get_multiple_series_lazy(exp_data, lambda individual_data:
        (lambda labelled_ts: [(label, is_pain, grove_gsr_12_bit_adc_processing(get_raw_arduino_data(individual_data, DATA_FIELD, ts - TIME_TO_PREV, TOTAL_TIME, global_sync_ts),
                                                                                 baseline_window=2,
                                                                                 required_length=REQUIRED_TIME,
                                                                                 resample_length=(ts - TIME_TO_PREV, INTERVAL_WIDTH, INTERVALS)))
            for label, is_pain, global_sync_ts, ts in labelled_ts])
            (get_labelled_pick_up_action_timestamp(individual_data, 
                list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], 
                get_trial_start_timestamps(individual_data)[start_trial_for_analysis:end_trial_for_analysis])), subjects)

    save_cache((all_gsr_with_labels_seen, all_gsr_with_labels_pick), CACHE_FILE_NAME)
else:
    all_gsr_with_labels_seen = all_gsr_with_labels[0]
    all_gsr_with_labels_pick = all_gsr_with_labels[1]

import numpy as np
import math

def generate_gsr_matrix(gsr_with_labels):
    matrices_by_pain = [[] for _ in range(len(pain_cond_idx))]
    pain_matrices_by_pain = [[] for _ in range(len(pain_cond_idx))]
    nopain_matrices_by_pain = [[] for _ in range(len(pain_cond_idx))]
    counted = 0
    total = 0
    for gsr_with_labels in gsr_with_labels:
        # subtract between pain choice mean and non-pain choice mean
        pain_choice_matrix = [[] for _ in range(len(pain_cond_idx))]
        nopain_choice_matrix = [[] for _ in range(len(pain_cond_idx))]
        for gsr in gsr_with_labels:
            if not gsr[2] is None:
                if gsr[1] == True:
                    if math.nan in gsr[2]:
                        print(gsr[2])
                    pain_choice_matrix[pain_cond_idx.index(gsr[0])].append(gsr[2])
                else:
                    nopain_choice_matrix[pain_cond_idx.index(gsr[0])].append(gsr[2])

        for i in range(len(pain_cond_idx)):
            try:
                total += 1
                pain_avg_signal = np.nanmean(np.stack(pain_choice_matrix[i], axis=0), axis=0)
                nopain_avg_signal = np.nanmean(np.stack(nopain_choice_matrix[i], axis=0), axis=0)
                matrices_by_pain[i].append(pain_avg_signal - nopain_avg_signal)
                pain_matrices_by_pain[i].append(pain_avg_signal)
                nopain_matrices_by_pain[i].append(nopain_avg_signal)
                counted += 1
            except ValueError:
                pain_matrices_by_pain[i].append([np.nan] * INTERVALS)
                nopain_matrices_by_pain[i].append([np.nan] * INTERVALS)
                matrices_by_pain[i].append([np.nan] * INTERVALS)

    print("Count rate:", counted, '/', total)
    return pain_matrices_by_pain, nopain_matrices_by_pain, matrices_by_pain

import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib.patches as mpatches

pain_matrices_by_pain, nopain_matrices_by_pain, _ = generate_gsr_matrix(all_gsr_with_labels_seen)

plt.figure(figsize=(20, 10))
axes = []
start_plot = 3
end_plot = 42
for i in range(len(pain_cond_idx)):
    axes.append(plt.subplot(2, 5, i + 1))
axes[0].set_ylabel('Fixation', fontsize=20)
for i, (pain_matrix, nopain_matrix) in enumerate(zip(pain_matrices_by_pain, nopain_matrices_by_pain)):
    if i != 0:
        axes[i].sharey(axes[0])
    pain_stacked_matrix = np.stack(pain_matrix, axis=0)
    axes[i].plot(np.arange(pain_stacked_matrix.shape[1])[start_plot:end_plot], np.nanmean(pain_stacked_matrix, axis=0)[start_plot:end_plot], c='red')
    axes[i].fill_between(np.arange(pain_stacked_matrix.shape[1])[start_plot:end_plot], np.nanmean(pain_stacked_matrix, axis=0)[start_plot:end_plot] + 1.96 * sem(pain_stacked_matrix, axis=0, nan_policy='omit')[start_plot:end_plot], np.nanmean(pain_stacked_matrix, axis=0)[start_plot:end_plot] - 1.96 * sem(pain_stacked_matrix, axis=0, nan_policy='omit')[start_plot:end_plot], facecolor='red', alpha=0.5)
    axes[i].set_ylim(-3, 3)
    nopain_stacked_matrix = np.stack(nopain_matrix, axis=0)
    axes[i].plot(np.arange(nopain_stacked_matrix.shape[1])[start_plot:end_plot], np.nanmean(nopain_stacked_matrix, axis=0)[start_plot:end_plot], c='black')
    axes[i].fill_between(np.arange(nopain_stacked_matrix.shape[1])[start_plot:end_plot], np.nanmean(nopain_stacked_matrix, axis=0)[start_plot:end_plot] + 1.96 * sem(nopain_stacked_matrix, axis=0, nan_policy='omit')[start_plot:end_plot], np.nanmean(nopain_stacked_matrix, axis=0)[start_plot:end_plot] - 1.96 * sem(nopain_stacked_matrix, axis=0, nan_policy='omit')[start_plot:end_plot], facecolor='black', alpha=0.5)
    axes[i].set_xticks([2, 12, 22, 32, 42], labels=["0", "1", "2", "3", "4"])
    if i == 4:
        axes[i].set_title(["0%", "25%", "50%", "75%", "100%"][i] + ' shock intensity', fontsize=20)
    else:
        axes[i].set_title(["0%", "25%", "50%", "75%", "100%"][i], fontsize=20)

    axes[i].axvline(x=int(TIME_TO_PREV / INTERVAL_WIDTH), c='black', linestyle='dashed')

pain_patch = mpatches.Patch(color='red', alpha=0.7, label='Painful fruit')
nopain_patch = mpatches.Patch(color='black', alpha=0.7, label='Non-painful fruit')
plt.gcf().legend(handles=[pain_patch, nopain_patch], fontsize=20, bbox_to_anchor=(0.92, 1.05))

pain_matrices_by_pain, nopain_matrices_by_pain, _ = generate_gsr_matrix(all_gsr_with_labels_pick)

axes = []
start_plot = 3
end_plot = 42
for i in range(len(pain_cond_idx)):
    axes.append(plt.subplot(2, 5, i + 1 + 5))
axes[0].set_ylabel('Pick-up (shock for painful fruit)', fontsize=20)
for i, (pain_matrix, nopain_matrix) in enumerate(zip(pain_matrices_by_pain, nopain_matrices_by_pain)):
    if i != 0:
        axes[i].sharey(axes[0])
    pain_stacked_matrix = np.stack(pain_matrix, axis=0)
    axes[i].plot(np.arange(pain_stacked_matrix.shape[1])[start_plot:end_plot], np.nanmean(pain_stacked_matrix, axis=0)[start_plot:end_plot], c='red')
    axes[i].fill_between(np.arange(pain_stacked_matrix.shape[1])[start_plot:end_plot], np.nanmean(pain_stacked_matrix, axis=0)[start_plot:end_plot] + 1.96 * sem(pain_stacked_matrix, axis=0, nan_policy='omit')[start_plot:end_plot], np.nanmean(pain_stacked_matrix, axis=0)[start_plot:end_plot] - 1.96 * sem(pain_stacked_matrix, axis=0, nan_policy='omit')[start_plot:end_plot], facecolor='red', alpha=0.5)
    axes[i].set_ylim(-3, 3)
    nopain_stacked_matrix = np.stack(nopain_matrix, axis=0)
    axes[i].plot(np.arange(nopain_stacked_matrix.shape[1])[start_plot:end_plot], np.nanmean(nopain_stacked_matrix, axis=0)[start_plot:end_plot], c='black')
    axes[i].fill_between(np.arange(nopain_stacked_matrix.shape[1])[start_plot:end_plot], np.nanmean(nopain_stacked_matrix, axis=0)[start_plot:end_plot] + 1.96 * sem(nopain_stacked_matrix, axis=0, nan_policy='omit')[start_plot:end_plot], np.nanmean(nopain_stacked_matrix, axis=0)[start_plot:end_plot] - 1.96 * sem(nopain_stacked_matrix, axis=0, nan_policy='omit')[start_plot:end_plot], facecolor='black', alpha=0.5)
    axes[i].set_xticks([2, 12, 22, 32, 42], labels=["0", "1", "2", "3", "4"])
    if i == 4:
        axes[i].set_title(["0%", "25%", "50%", "75%", "100%"][i] + ' shock intensity', fontsize=20)
    else:
        axes[i].set_title(["0%", "25%", "50%", "75%", "100%"][i], fontsize=20)
    axes[i].axvline(x=int(TIME_TO_PREV / INTERVAL_WIDTH), c='black', linestyle='dashed')

plt.gcf().text(0.07, 0.2, 'Skin conductance change from baseline (\u03bcS)', fontsize=20, rotation = 90)
plt.gcf().text(0.4, 0.05, 'Time after events (seconds)', fontsize=20)

plt.savefig('figures/PUB/gsr.svg', bbox_inches='tight')