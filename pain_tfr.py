# Copyright (c) 2026 Shuangyi Tong <shuangyi.tong@ndcn.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import numpy as np

import core.utils
core.utils.verbose = True

from core.utils import save_cache, load_cache

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_NAME, UNITY_DEVICE_ID, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME
from core.experiment_data import set_expt
from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series

import mne
from mne.stats import permutation_cluster_1samp_test

import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cl

def to_unix_timestamp(dt_obj):
    return time.mktime(dt_obj.timetuple())*1e3 + dt_obj.microsecond/1e3

EXPT_NAME = 'Expt4'
HIGH_PASS = 1 # change this to adjust high pass

set_expt(EXPT_NAME)

start_trials_for_analysis = 6
end_trials_for_analysis = 24

event_order = ["LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic", "NoPainNoPressure", "NoPainTonic"]

CACHE = 'PAIN_EPOCH'
epochs_list = None #load_cache(CACHE)
if epochs_list == None:
    exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME])
    subjects = list(exp_data.keys())
    print(subjects)
    reject_criteria = dict(eeg=200e-6)
    epochs_list = get_multiple_series(exp_data, 
                lambda individual_data: 
                    (lambda raw, event_array, event_map, event_inverse_map: mne.Epochs(
                        raw,
                        np.array([[x, y, event_order.index(event_inverse_map[z])] for x, y, z in event_array if z in event_inverse_map.keys()]),
                        event_map,
                        event_repeated='drop',
                        preload=True,
                        reject=reject_criteria, 
                        tmin=-0.55,
                        picks=['Cz'],
                        tmax=0.95,
                        on_missing='warn'))
                    (individual_data['eeg_clean'].set_eeg_reference('average').filter(HIGH_PASS, None).notch_filter(freqs=50, notch_widths=4),
                    mne.events_from_annotations(individual_data['eeg_clean'])[0],
                    { k: event_order.index(k) for k, _ in mne.events_from_annotations(individual_data['eeg_clean'])[1].items() if k in event_order},
                    { v: k for k, v in mne.events_from_annotations(individual_data['eeg_clean'])[1].items() if k in event_order})
                if 'eeg_clean' in individual_data.keys() else None, subjects)

    epochs_list = list(filter(lambda x: x, epochs_list))
    save_cache(epochs_list, CACHE)

drop_log_rate = [epochs.drop_log_stats() for epochs in epochs_list]
print(drop_log_rate)
print('average drop rate:', np.mean(drop_log_rate))

def tfr_for_epochs(condition, ax, t):
    # 1. Define your TFR parameters
    freqs = np.geomspace(4, 100, num=40)
    n_cycles = freqs / 2.

    all_tfr = []

    # Loop through each subject's epochs
    for epochs in epochs_list:
        # Compute TFR for this subject's condition
        tfr = mne.time_frequency.tfr_morlet(
            epochs[condition], 
            freqs=freqs, 
            n_cycles=n_cycles, 
            average=True, 
            return_itc=False,
        )
        
        # 3. Apply baseline correction PER SUBJECT
        # This is critical for TFRs to account for individual power differences
        tfr.apply_baseline(baseline=(-0.3, 0), mode='logratio')
        
        all_tfr.append(tfr.crop(tmin=-0.3, tmax=0.7))

    mne.grand_average(all_tfr).plot(axes=ax, picks='Cz', tmin=-0.3, tmax=0.7, vlim=(-0.7, 0.7), colorbar=False, show=False, cmap='jet')
    ax.set_title(t, fontsize=24)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Frequency (Hz)', fontsize=16)
    ax.tick_params(labelsize=16)

    return all_tfr

def permutation_cluster(tfr_list1, tfr_list2, ax, t):
    X = []
    for h, l in zip(tfr_list1, tfr_list2):
        # Difference for a single channel (Cz)
        diff = h.copy().pick('Cz').data[0] - l.copy().pick('Cz').data[0]
        X.append(diff)

    diff_tfr = np.array(X)
    print(diff_tfr.shape)

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        diff_tfr, 
        threshold=None, 
        n_permutations=10000,
        tail=0, # two-tailed test
        out_type='mask',
        seed=0
    )

    times = np.linspace(-0.3, 0.7, diff_tfr.shape[2])
    frequencies = np.geomspace(4, 100, num=40)

    # 1. Plot the actual TFR data
    im = ax.pcolormesh(times, frequencies, T_obs, cmap='RdBu_r', shading='auto', vmin=-7, vmax=7)
    ax.set_yscale('log') # Since we used geomspace
    ax.set_yticks([4, 8, 13, 30, 70, 100])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # 3. Overlay the Significant Clusters
    for i, cluster_p in enumerate(cluster_p_values):
        if cluster_p <= 0.05:
            print('there is a cluster with cluster p', cluster_p)
            # 'clusters' is a list of boolean masks (freqs x times)
            mask = clusters[i]
            
            # Draw a contour around the cluster
            ax.contour(times, frequencies, mask, colors='black', linewidths=3)
            
    ax.set_title(t, fontsize=24)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Frequency (Hz)', fontsize=16)
    ax.tick_params(labelsize=16)

fig, axes = plt.subplots(2, 2, figsize=(20, 20))

axes_flat = axes.flatten()

tfr_high_nopressure = tfr_for_epochs("HighNoPressure", axes_flat[0], "High phasic pain & No tonic pain")
tfr_low_nopressure = tfr_for_epochs("LowNoPressure", axes_flat[1], "Low phasic pain & No tonic pain")
tfr_high_tonic = tfr_for_epochs("HighTonic", axes_flat[2], "High phasic pain & With tonic pain")
tfr_low_tonic = tfr_for_epochs("LowTonic", axes_flat[3], "Low phasic pain & With tonic pain")
cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
cbar_ax.tick_params(labelsize=16)
plt.colorbar(cm.ScalarMappable(norm=cl.Normalize(vmin=-0.7, vmax=0.7), cmap=plt.get_cmap('jet')), cax=cbar_ax)
fig.suptitle('Induced Oscillatory Response to Phasic Pain Stimulus at Electrode Cz ($\log_{10}(\mu V^2)$)', fontsize=24)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 20))

axes_flat = axes.flatten()
permutation_cluster(tfr_high_nopressure, tfr_low_nopressure, axes_flat[0], 'High phasic pain - Low phasic pain (No tonic pain)')
permutation_cluster(tfr_high_nopressure, tfr_high_tonic, axes_flat[1], 'No tonic pain - With tonic pain (High phasic pain)')
permutation_cluster(tfr_high_tonic, tfr_low_tonic, axes_flat[2], 'High phasic pain - Low phasic pain (With tonic pain)')
permutation_cluster(tfr_low_nopressure, tfr_low_tonic, axes_flat[3], 'No tonic pain - With tonic pain (Low phasic pain)')
cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
cbar_ax.tick_params(labelsize=16)
cbar_ax.set_title('$t$ statistics', fontsize=24)
plt.colorbar(cm.ScalarMappable(norm=cl.Normalize(vmin=-7, vmax=7), cmap=plt.get_cmap('RdBu_r')), cax=cbar_ax)
fig.suptitle('Induced Oscillatory Response to Phasic Pain Stimulus at Electrode Cz (Cluster-based Permutation Test)', fontsize=24)
plt.show()