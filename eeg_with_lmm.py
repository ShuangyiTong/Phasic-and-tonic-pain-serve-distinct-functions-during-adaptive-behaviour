# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import model_validation
import realtime_model

import sys
import os
import json
import pandas as pd
import math
import numpy as np
import time
import logging

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME, UNITY_DEVICE_NAME, save_cache, load_cache
from core.experiment_data import set_expt
from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_lazy_subject_indexed, get_multiple_series_lazy

from core.individual_subject import filter_band_psds, generate_eeg_forward_model, get_natural_number_indexed_value_and_fill_gap_by_zero

logging.getLogger("mne").setLevel(logging.WARNING) # ask mne welch to shut up for reporting effective window size INFO

DELTA = (1, 4)
THETA = (4, 8)
ALPHA = (8, 13)
BETA = (13, 30)
GAMMA = (30, 101)
LOW_GAMMA = (30, 46)
HIGH_GAMMA = (55, 91)
bands = {#'Delta (1-4 Hz)': DELTA,
         'Theta (4-7 Hz)': THETA,
         'Alpha (8-12 Hz)': ALPHA, 
         'Beta (13-29 Hz)': BETA,
         'Gamma (30-100 Hz)': GAMMA,
         #'Low Gamma (30-45 Hz)': LOW_GAMMA,
         #'High Gamma (55-90 Hz)': HIGH_GAMMA
        }

SEPARATE_VIGOUR = False
if SEPARATE_VIGOUR:
    tonic_file = 'model_fitting_results/separate_vigour/FalseTrue_fitted.json'
    no_tonic_file = 'model_fitting_results/separate_vigour/TrueFalse_fitted.json'
else:
    tonic_file = 'model_fitting_results/single_vigour/FalseTrue_fitted.json'
    no_tonic_file = 'model_fitting_results/single_vigour/TrueFalse_fitted.json'

Expt_name = sys.argv[1]
set_expt(Expt_name)
mode = sys.argv[2] # 'surface' | 'volume' | 'surface_source'
beamformer_method = 'lcmv' # 'dics' or 'lcmv'
surface_source_method = 'sLORETA' # “MNE” | “dSPM” | “sLORETA” | “eLORETA”
# 'epoch' for filter calculated based on individual fixation epochs and and 'block' for entire block. 'epoch' not implemented for 'dics'
beamformer_filter_based_data = 'epoch'
lcmv_weight_norm = 'nai'
spectral_method = 'morlet' # for dics only
surface_power_method = 'welch' # 'welch' | 'fft' | 'morlet' | 'multitaper', also used for lcmv
surface_power_transform_before_lm = lambda x: x
surface_power_in_db = False
surface_exclusion_by_zscore = 3 # False / 0 for no exclusion, a positive number to define max zscore by band power by activity
surface_regression_model = 'lmm' # 'lmm' | 'glm' | 'glmm'
surface_sample_range = (0, 250)
surface_sample_baseline_range = None #(-250, 0) # None for no baseline, not available for morlet
minimum_norm_noise_epoch_range = ('adhoc', 250)
covariance_method = 'empirical'
surface_source_band_pass_filter = None
volume_early_reject = True
surface_exclusion_by_other_band = None # if enabled, must already run the None version to generate correct cache in temp folder

MORLET_LOW_FREQ = 4
MORLET_HIGH_FREQ = 100
FWHM_MINIMUM_FACTOR = 2

all_tonic_fitted_params = None
all_no_tonic_fitted_params = None

from core.individual_subject import get_fruit_position_map, get_abstract_action_v2, get_trial_start_timestamps

containmination_reject_epoch_name_generator = lambda band_name, subject: 'remove_epoch_' + band_name + '_' + str(SEPARATE_VIGOUR) + '_' + subject

def global_sim_epoched_quantity_dump(tonic_fitted_params, no_tonic_fitted_params, pineapple_maps, pain_conditions, behavioural_data):
    accs = []
    dump_data = []
    for x in range((realtime_model.end_trial_for_analysis if realtime_model.end_trial_for_analysis else 0) - realtime_model.start_trial_for_analysis):
        is_tonic = True if "Tonic" in pain_conditions[x][1] else False
        if is_tonic:
            realtime_model.NO_CONDITION_1 = False
            realtime_model.NO_CONDITION_2 = True
        else:
            realtime_model.NO_CONDITION_1 = True
            realtime_model.NO_CONDITION_2 = False

        fitted_params = tonic_fitted_params if is_tonic else no_tonic_fitted_params
        acc, dump = realtime_model.in_block_simulation(model_validation.convert_cpp_params_to_python(fitted_params,
                                                                                                     duplicate_vigour_for_phasic_pain_cond=(not SEPARATE_VIGOUR)), 
                                                       pineapple_maps[x], pain_conditions[x], *(behavioural_data[x]),
                                                       visualization_verbose=False, block_id=x, dump_key_epochs=True)
        accs.append(acc)
        dump_data.append(dump)

    if abs(np.mean(accs) - (tonic_fitted_params[-1] + no_tonic_fitted_params[-1]) / 2) > model_validation.ACCURACY_EPSILON:
        print("Model fitting check failed - CPP:", fitted_params[-1], " >>>>>>>>> Python:", np.nanmean(accs))
        return False, dump_data
    else:
        print("Pass - CPP:", (tonic_fitted_params[-1] + no_tonic_fitted_params[-1]) / 2, " ======== Python:", np.nanmean(accs))

    return True, dump_data

def get_key_quantities_for_epoch_data(individual_data, subject):
    tonic_fitted_params = all_tonic_fitted_params[subject]
    no_tonic_fitted_params = all_no_tonic_fitted_params[subject]

    pineapple_maps = get_fruit_position_map(individual_data)[realtime_model.start_trial_for_analysis:realtime_model.end_trial_for_analysis]
    behavioural_data = [get_abstract_action_v2(individual_data, ts) for ts in get_trial_start_timestamps(individual_data)][realtime_model.start_trial_for_analysis:
                                                                                                                           realtime_model.end_trial_for_analysis]
    pain_conditions = realtime_model.get_pain_conditions(individual_data)
    print(len(pineapple_maps), len(behavioural_data))
    print(pain_conditions)

    pressures = get_natural_number_indexed_value_and_fill_gap_by_zero(individual_data, 'pressure', 24)[realtime_model.start_trial_for_analysis:
                                                                                                       realtime_model.end_trial_for_analysis]
    pressure_ratings = get_natural_number_indexed_value_and_fill_gap_by_zero(individual_data, 'pressure_ratings', 24)[realtime_model.start_trial_for_analysis:
                                                                                                                      realtime_model.end_trial_for_analysis]

    assert(len(pressures) == len(pressure_ratings) == len(pineapple_maps) == len(behavioural_data) == len(pain_conditions))

    passed, dump_data = global_sim_epoched_quantity_dump(tonic_fitted_params, no_tonic_fitted_params, pineapple_maps, pain_conditions, behavioural_data)

    if not passed:
        raise RuntimeError('Not passed, can\'t proceed')
    else:
        assert(len(pressures) == len(pressure_ratings) == len(dump_data))
        return dump_data, pressures, pressure_ratings

all_dump_data = None #load_cache('key_epoch_dump')

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB14", "SUB20"
                                                                                 ],#"SUB9", "SUB18"],  # left handed
                                exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION], lazy_closure=True)
subjects = list(exp_data.keys())

fwd = generate_eeg_forward_model(exp_data[subjects[0]](subjects[0])['eeg_clean'], volumetric_source=(False if mode == 'surface_source' or beamformer_method == 'lcmv' else True))#, check_alignment=True, alignment_info=exp_data[subjects[0]](subjects[0])['eeg_clean'].info)

if all_dump_data == None:
    with open(tonic_file) as f:
        all_tonic_fitted_params = json.load(f)

    with open(no_tonic_file) as f:
        all_no_tonic_fitted_params = json.load(f)

    all_dump_data = get_multiple_series_lazy_subject_indexed(exp_data, get_key_quantities_for_epoch_data, subjects)

    all_dump_data = { subject: dump_data for subject, dump_data in zip(subjects, all_dump_data)}

    save_cache(all_dump_data, 'key_epoch_dump')

import mne
import numpy as np
from scipy.fft import fft, fftfreq, rfftfreq
from scipy.signal.windows import hamming

def check_trial_marker_not_found(eeg_ts, markers, threshold=0):
    for marker in markers:
        if abs(eeg_ts - marker) <= threshold:
            return False
        
    return True

def fft_with_window(raw_data_segment, N, output='magnitude'):
    # compute fast fourier transform with hamming window
    # https://docs.scipy.org/doc/scipy/tutorial/fft.html
    T = 1.0 / 500.0 # 500 Hz
    w = hamming(N)
    ywf = fft(raw_data_segment * w)
    freqs = fftfreq(N, T)[1:N//2]
    psds = 2.0/N * np.abs(ywf[1:N//2])
    if output == 'power':
        psds *= psds # to power, equivalent to welch modulo constant.
    elif output == 'magnitude':
        pass
    else:
        raise ValueError('Unkown output type: ' + output) 

    return psds, freqs

def multitaper_freqs(fmin, fmax, n_fft, sfreq):
    # Copied from MNE 1.7.0 source mne.time_frequency.csd:1014-1017
    orig_frequencies = rfftfreq(n_fft, 1.0 / sfreq)
    freq_mask = (orig_frequencies > fmin) & (orig_frequencies < fmax)
    frequencies = orig_frequencies[freq_mask]

    return frequencies

subject_id = []
op_cost = []
phasic_pain = []
vigour_cost = []
choice_value = []
head_movement = []
tonic = []
pressure_val = []
pressure_rating_val = []
pressure_val_tonic_only = []
pressure_rating_val_tonic_only = []
vigour_constant = []
beta_dot_distance_times_C_v = []
beta_distance = []
spectral_power = []
electrode_name = []
epoch_id = []
epoch_counter = 0
freqs_for_all = None
info_for_all = None
if spectral_method == 'morlet':
    dics_freqs = [np.logspace(np.log10(band[0]), np.log10(band[1]), 20, endpoint=False) for _, band in bands.items()]
    csd_func = mne.time_frequency.csd_array_morlet
    tfr_func = mne.time_frequency.tfr_array_morlet
elif spectral_method == 'multitaper':
    dics_freqs = [multitaper_freqs(band[0], band[1], 500, 500) for _, band in bands.items()]
    csd_func = mne.time_frequency.csd_array_multitaper
    tfr_func = mne.time_frequency.tfr_array_multitaper

subject_id_bf = []
op_cost_bf = []
phasic_pain_bf = []
vigour_cost_bf = []
choice_value_bf = []
head_movement_bf = []
tonic_bf = []
pressure_val_bf = []
pressure_rating_val_bf = []
pressure_val_tonic_only_bf = []
pressure_rating_val_tonic_only_bf = []
vigour_constant_bf = []
beta_dot_distance_times_C_v_bf = []
beta_distance_bf = []
spectral_power_bf = []
dipole_idx = []
vol_estimate_container = None

total_epochs = sum([sum([len(key_epochs[1]['timestamp_ms']) for key_epochs in all_dump_data[subject][0]]) for subject in subjects])
total_epoch_counter = 0

def calculate_psd(data):
    if surface_power_method == 'welch':
        psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, n_fft=len(data))
    elif surface_power_method == 'multitaper':
        psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=500)
    elif surface_power_method == 'fft':
        psds, freqs = fft_with_window(data, len(data), output='magnitude')
    else:
        raise ValueError('No such method ' + surface_power_method)
    return psds, freqs

for subject in subjects:
    epoch_counter = 0 # reset for each subject to ensure consistency when applying from surface exclusion to volume exclusion
    individual_data = exp_data[subject](subject)
    raw_data = individual_data['eeg_clean'].set_eeg_reference('average').notch_filter(freqs=50, notch_widths=4)
    mne_format_events = mne.events_from_annotations(individual_data['eeg_clean'])
    key_epochs_data = all_dump_data[subject][0]
    pressures = all_dump_data[subject][1]
    pressure_ratings = all_dump_data[subject][2]
    if volume_early_reject and mode in ['surface_source', 'volume']:
        reject_epoch_lists = []
        for i, (band_name, band) in enumerate(bands.items()):
            reject_epoch_lists.append(load_cache(containmination_reject_epoch_name_generator(band_name, subject)))
    if info_for_all is None:
        info_for_all = raw_data.info
    else:
        assert(info_for_all['ch_names'] == raw_data.info['ch_names'])

    # check start timestamp alignment
    trial_start_event_list = ['trial_start']
    old_event_inverse_map = { v: k for k, v in mne_format_events[1].items() if k in trial_start_event_list}
    trial_start_event_array = np.array([[x, y, trial_start_event_list.index(old_event_inverse_map[z])] for x, y, z in mne_format_events[0] if z in old_event_inverse_map.keys()], dtype=np.int32)

    trial_start_ts = get_trial_start_timestamps(individual_data)[realtime_model.start_trial_for_analysis:
                                                                 realtime_model.end_trial_for_analysis]
    global_start_ts = trial_start_ts[0]
    total_epochs_this_subject = sum([len(key_epochs[1]['timestamp_ms']) for key_epochs in key_epochs_data])
    for (start_ts, key_epochs), pressure, pressure_rating in zip(key_epochs_data, pressures, pressure_ratings):
        # we should always have the trial_start marker on the sample index by calculating start_ts - global_start_ts + 1s (500 samples) because in preproccess_for_eeglab:122 we have -1 second for the first global time stamp
        assert(check_trial_marker_not_found(int((start_ts - global_start_ts) / 2 + 500),
                                                trial_start_event_array.T[0]))
        
        if mode == 'volume':
            if beamformer_method == 'lcmv':
                block_filters = []
                raw_data_bands_copy_with_projection = [] 
                if beamformer_filter_based_data == 'block':
                    for i, (band_name, band) in enumerate(bands.items()):
                        raw_data_band_copy = raw_data.copy().filter(band[0], band[1])
                        block_cov = mne.compute_raw_covariance(raw_data_band_copy, tmin=(start_ts - global_start_ts) / 1000.0 + 1,
                                                            tmax=(start_ts - global_start_ts) / 1000.0 + 1 + 60, method="empirical")
                        block_filters.append(mne.beamformer.make_lcmv(
                                        raw_data.info,
                                        fwd,
                                        block_cov,
                                        noise_cov=None,
                                        weight_norm=lcmv_weight_norm,
                                        rank=None,
                                    ))
                        raw_data_bands_copy_with_projection.append(raw_data_band_copy.set_eeg_reference(projection=True))
                elif beamformer_filter_based_data == 'epoch':
                    for i, (band_name, band) in enumerate(bands.items()):
                        raw_data_band_copy = raw_data.copy().filter(band[0], band[1])
                        if volume_early_reject:
                            events_array = [[int((key_epoch_ts - global_start_ts) / 2) + 500, 0, 1] for epoch_idx, key_epoch_ts in enumerate(key_epochs['timestamp_ms']) if epoch_idx + epoch_counter not in set(reject_epoch_lists[i])]
                        else:
                            events_array = [[int((key_epoch_ts - global_start_ts) / 2) + 500, 0, 1] for key_epoch_ts in key_epochs['timestamp_ms']]
                        key_epochs_eeg = mne.Epochs(raw_data_band_copy, events=np.array(events_array), event_id=1,
                                                    tmin=surface_sample_range[0] / 500.0, tmax=surface_sample_range[1] / 500.0, baseline=(0, 0))
                        block_cov = mne.compute_covariance(key_epochs_eeg, method=covariance_method)
                        block_filters.append(mne.beamformer.make_lcmv(
                                raw_data.info,
                                fwd,
                                block_cov,
                                noise_cov=None,
                                weight_norm=lcmv_weight_norm,
                                rank=None,
                            ))
                        raw_data_bands_copy_with_projection.append(raw_data_band_copy.set_eeg_reference(projection=True))
                else:
                    raise ValueError('beamformer_filter_based_data=' + beamformer_filter_based_data + ' not recognised.')
            elif beamformer_method == 'dics':
                block_stcs = []
                for i, dics_freq in enumerate(dics_freqs):
                    block_data = raw_data.get_data(tmin=(start_ts - global_start_ts) / 1000.0 + 1 - 1, # raw data has the original sample indices, 1s offset confirmed by above checks
                                                   tmax=(start_ts - global_start_ts) / 1000.0 + 1 + 60 + 1) * 1e6 # to uV, added 1s in case of start point < 250
                    if spectral_method == 'morlet':
                        block_csd = csd_func(np.array([block_data]), sfreq=500, 
                                            frequencies=dics_freq, ch_names=raw_data.info['ch_names'])
                    elif spectral_method == 'multitaper':
                        block_csd = csd_func(np.array([block_data]), sfreq=500, ch_names=raw_data.info['ch_names'],
                                            fmin=[band for _, band in bands.items()][i][0],
                                            fmax=[band for _, band in bands.items()][i][1])
                    # Only average frequencies in the filter for apply_dics_csd, not volumetric data
                    # block_csd = block_csd.mean()
                    block_filters = mne.beamformer.make_dics(
                        raw_data.info,
                        fwd,
                        block_csd,
                        pick_ori="max-power",
                        reduce_rank=True,
                        real_filter=True,
                    )
                    del block_csd
                    desired_fwhm = (1 / dics_freq) * FWHM_MINIMUM_FACTOR
                    n_cycles = desired_fwhm * np.pi * np.array(dics_freq) / np.sqrt(2 * np.log(2))
                    block_tfr = tfr_func(np.array([block_data]), sfreq=500, freqs=dics_freq, output='complex', zero_mean=True, n_cycles=n_cycles)
                    assert (block_tfr.shape[3] == block_data.shape[1])
                    block_tfr = mne.time_frequency.EpochsTFRArray(raw_data.info, data=block_tfr, times=np.linspace(-1, 61, block_data.shape[1], endpoint=False), freqs=dics_freq)
                    block_stc = mne.beamformer.apply_dics_tfr_epochs(block_tfr, block_filters)[0]
                    assert(fwd['nsource'] == block_stc[0].data.shape[0])
                    assert(block_stc[0].data.shape[1] == block_data.shape[1])
                    del block_tfr
                    del block_filters
                    data = np.zeros((fwd["nsource"], block_data.shape[1]))
                    for stc in block_stc:
                        data += (stc.data * np.conj(stc.data)).real
                    block_stc[0].data = data / len(dics_freq)
                    block_stc = block_stc[0]
                    block_stcs.append(block_stc)
        elif mode == 'surface_source':
            inverse_oprators = []
            raw_data_bands_copy_with_projection = [] 
            for i, (band_name, band) in enumerate(bands.items()):
                if volume_early_reject:
                    events_array = [[int((key_epoch_ts - global_start_ts) / 2) + 500, 0, 1] for epoch_idx, key_epoch_ts in enumerate(key_epochs['timestamp_ms']) if epoch_idx + epoch_counter not in set(reject_epoch_lists[i])]
                else:
                    events_array = [[int((key_epoch_ts - global_start_ts) / 2) + 500, 0, 1] for key_epoch_ts in key_epochs['timestamp_ms']]
                raw_data_band_copy = raw_data.copy().filter(band[0], band[1])
                if minimum_norm_noise_epoch_range[0] == 'adhoc':
                    noise_cov = mne.make_ad_hoc_cov(raw_data.info)
                else:
                    noise_epochs_eeg = mne.Epochs(raw_data_band_copy, events=np.array(events_array), event_id=1,
                                                tmin=minimum_norm_noise_epoch_range[0] / 500.0, tmax=minimum_norm_noise_epoch_range[1] / 500.0, baseline=(0, 0))
                    noise_cov = mne.compute_covariance(noise_epochs_eeg, method=covariance_method, rank=None)
                inverse_operator = mne.minimum_norm.make_inverse_operator(
                    raw_data.info, fwd, noise_cov=noise_cov, loose=0.2, depth=2 # MNE recommend for EEG use 2-5
                )
                inverse_oprators.append(inverse_operator)
                raw_data_bands_copy_with_projection.append(raw_data_band_copy.set_eeg_reference(projection=True))

        elif mode == 'surface' and surface_power_method == 'morlet':
            block_data = raw_data.get_data(tmin=(start_ts - global_start_ts) / 1000.0 + 1, # raw data has the original sample indices, 1s offset confirmed by above checks
                                           tmax=(start_ts - global_start_ts) / 1000.0 + 1 + 60)  * 1e6 # to uV unit
            
            freqs = np.logspace(np.log10(4), np.log10(100), 100)
            freqs_for_all = freqs
            desired_fwhm = (1 / freqs) * FWHM_MINIMUM_FACTOR
            n_cycles = desired_fwhm * np.pi * np.array(freqs) / np.sqrt(2 * np.log(2))
            # print('n_cycles: ', n_cycles)
            block_tfr = mne.time_frequency.tfr_array_morlet(np.array([block_data]), sfreq=500, freqs=freqs, output='power', zero_mean=True, n_cycles=n_cycles)
            assert (block_tfr.shape[3] == block_data.shape[1])

        for epoch_idx, key_epoch_ts in enumerate(key_epochs['timestamp_ms']):
            samples_idx = int((key_epoch_ts - global_start_ts) / 2) + 500
            if mode == 'surface':
                if surface_power_method != 'morlet':
                    raw_data_seg = raw_data.get_data(start=samples_idx + surface_sample_range[0], stop=samples_idx + surface_sample_range[1]) * 1e6 # to uV unit
                    if surface_sample_baseline_range:
                        raw_data_seg_baseline = raw_data.get_data(start=samples_idx + surface_sample_baseline_range[0], stop=samples_idx + surface_sample_baseline_range[1]) * 1e6 # to uV unit
                for electrodes_idx, ch_name in enumerate(raw_data.info["ch_names"]):
                    if surface_power_method == 'morlet':
                        start_sample = int((key_epoch_ts - start_ts) / 2)
                        psds = block_tfr[0, electrodes_idx,:, start_sample + surface_sample_range[0]:start_sample + surface_sample_range[1]]
                        assert(psds.shape[0] == len(freqs_for_all) and len(psds.shape) == 2)
                        psds = np.mean(psds, axis=1) # average over 1s time period, keep freqs
                    else:
                        raw_data_electrode_seg = raw_data_seg[electrodes_idx]
                        psds, freqs = calculate_psd(raw_data_electrode_seg)
                        if surface_sample_baseline_range:
                            psds_baseline, freqs_baseline = calculate_psd(raw_data_seg_baseline[electrodes_idx])
                            assert(np.array_equal(freqs, freqs_baseline))
                            psds = psds - psds_baseline

                        if freqs_for_all is None:
                            freqs_for_all = freqs
                        else:
                            assert(np.array_equal(freqs_for_all, freqs))
                    
                    subject_id.append(subject)
                    op_cost.append(key_epochs['op_cost'][epoch_idx])
                    phasic_pain.append(key_epochs['phasic_pain'][epoch_idx])
                    vigour_cost.append(key_epochs['vigour_cost'][epoch_idx])
                    choice_value.append(key_epochs['choice_value'][epoch_idx])
                    vigour_constant.append(key_epochs['vigour_constant'][epoch_idx])
                    beta_dot_distance_times_C_v.append(key_epochs['beta_dot_distance_times_C_v'][epoch_idx])
                    beta_distance.append(key_epochs['beta_distance'][epoch_idx])
                    head_movement.append(key_epochs['head_movement'][epoch_idx])
                    tonic.append(key_epochs['tonic'][epoch_idx])
                    pressure_val.append(pressure)
                    pressure_rating_val.append(pressure_rating)
                    pressure_val_tonic_only.append((pressure if pressure > 0 else math.nan))
                    pressure_rating_val_tonic_only.append((pressure_rating if pressure_rating > 0 else math.nan))
                    spectral_power.append(psds)
                    electrode_name.append(ch_name)
                    epoch_id.append(epoch_counter)

            elif mode in ['volume', 'surface_source']:
                if beamformer_method == 'lcmv' or mode == 'surface_source':
                    key_epoch_stcs = []
                    if mode == 'volume':
                        for i, (band_name, band) in enumerate(bands.items()):
                            key_epoch_stc = mne.beamformer.apply_lcmv_raw(raw_data_bands_copy_with_projection[i], block_filters[i], 
                                                                        start=samples_idx + surface_sample_range[0], stop=samples_idx + surface_sample_range[1])
                            assert(fwd['nsource'] == key_epoch_stc.data.shape[0])
                            key_epoch_stcs.append(key_epoch_stc)
                    else: # must in surface_source mode
                        for i, (band_name, band) in enumerate(bands.items()):
                            snr = 1.0 # 3 for averaged and 1 for non-averaged data
                            lambda2 = 1.0 / snr**2
                            key_epoch_stc = mne.minimum_norm.apply_inverse_raw(raw_data_bands_copy_with_projection[i], inverse_oprators[i], lambda2=lambda2, method=surface_source_method,
                                                                               start=samples_idx + surface_sample_range[0], stop=samples_idx + surface_sample_range[1])
                            key_epoch_stcs.append(key_epoch_stc)

                    for dipole in range(fwd['nsource']):
                        print('processing dipole: ', dipole, end='\r')
                        agg_psds = []
                        for i, (band_name, band) in enumerate(bands.items()):
                            key_epoch_stc = key_epoch_stcs[i]
                            raw_data_dipole_seg = key_epoch_stc.data[dipole]
                            psds, freqs = calculate_psd(raw_data_dipole_seg)
                            if mode == 'surface_source' and surface_source_band_pass_filter == None:
                                new_psds, _ = filter_band_psds(psds, freqs, (0, 500)) # don't filter psd
                            else:
                                new_psds, _ = filter_band_psds(psds, freqs, band)
                            agg_psd = np.mean(new_psds)
                            agg_psds.append(agg_psd)

                        subject_id_bf.append(subject)
                        op_cost_bf.append(key_epochs['op_cost'][epoch_idx])
                        phasic_pain_bf.append(key_epochs['phasic_pain'][epoch_idx])
                        vigour_cost_bf.append(key_epochs['vigour_cost'][epoch_idx])
                        choice_value_bf.append(key_epochs['choice_value'][epoch_idx])
                        vigour_constant_bf.append(key_epochs['vigour_constant'][epoch_idx])
                        beta_dot_distance_times_C_v_bf.append(key_epochs['beta_dot_distance_times_C_v'][epoch_idx])
                        head_movement_bf.append(key_epochs['head_movement'][epoch_idx])
                        beta_distance_bf.append(key_epochs['beta_distance'][epoch_idx])
                        tonic_bf.append(key_epochs['tonic'][epoch_idx])
                        pressure_val_bf.append(pressure)
                        pressure_rating_val_bf.append(pressure_rating)
                        pressure_val_tonic_only_bf.append((pressure if pressure > 0 else math.nan))
                        pressure_rating_val_tonic_only_bf.append((pressure_rating if pressure_rating > 0 else math.nan))
                        spectral_power_bf.append(agg_psds)
                        dipole_idx.append(dipole)
                        epoch_id.append(epoch_counter)


                elif beamformer_method == 'dics':
                    start_sample = int((key_epoch_ts - start_ts) / 2) + 500 # added 1s in case of start point < 250
                    for dipole in range(fwd['nsource']):
                        subject_id_bf.append(subject)
                        op_cost_bf.append(key_epochs['op_cost'][epoch_idx])
                        phasic_pain_bf.append(key_epochs['phasic_pain'][epoch_idx])
                        vigour_cost_bf.append(key_epochs['vigour_cost'][epoch_idx])
                        choice_value_bf.append(key_epochs['choice_value'][epoch_idx])
                        vigour_constant_bf.append(key_epochs['vigour_constant'][epoch_idx])
                        beta_dot_distance_times_C_v_bf.append(key_epochs['beta_dot_distance_times_C_v'][epoch_idx])
                        head_movement_bf.append(key_epochs['head_movement'][epoch_idx])
                        tonic_bf.append(key_epochs['tonic'][epoch_idx])
                        spectral_power_bf.append([np.mean(block_stc.data[dipole][start_sample - 250:start_sample + 250]) 
                                                  for block_stc in block_stcs])
                        dipole_idx.append(dipole)

                if not vol_estimate_container:
                    if beamformer_method == 'dics':
                        vol_estimate_container = block_stc
                    else:
                        vol_estimate_container = key_epoch_stc
            
            epoch_counter += 1
            total_epoch_counter += 1
            print('complete epoch ' + str(epoch_counter) + '/' + str(total_epochs_this_subject) + ' for subject ' + subject + ' total epochs: ' + str(total_epochs) + ' >>> ' + str(total_epoch_counter / total_epochs * 100) + '%')
        if mode == 'volume' and beamformer_method == 'lcmv':
            del block_filters
    del raw_data

assert(len(subject_id) == len(op_cost) == len(phasic_pain) == len(vigour_cost) == len(choice_value) == len(spectral_power) == len(electrode_name) == len(head_movement) == len(vigour_constant) == len(beta_dot_distance_times_C_v) == len(beta_distance))
assert(len(subject_id_bf) == len(op_cost_bf) == len(phasic_pain_bf) == len(vigour_cost_bf) == len(choice_value_bf) == len(spectral_power_bf) == len(dipole_idx) == len(head_movement_bf) == len(vigour_constant_bf) == len(beta_dot_distance_times_C_v_bf) == len(beta_distance_bf))
import pandas as pd

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
generics = importr('generics')

def extract_mixed_model_stats(m, look_for, model='lmer'):
    if model == 'lmer':
        offset = 0
    elif model == 'glmer':
        # glmm don't provide degree of freedom, its statistics is not t or z distributed
        # https://stats.stackexchange.com/questions/281974/how-can-i-obtain-z-values-instead-of-t-values-in-linear-mixed-effect-model-lmer
        offset = 1
    else:
        raise ValueError('model type ' + model + ' not recognised')
    p_val = None
    summary = str(base.summary(m))
    # with open(str(time.time()), 'w') as f:
    #     f.write(summary)
    lines = summary.splitlines()
    for line in lines:
        line_elements = line.split()
        if len(line_elements) >= 6 - offset:
            if line_elements[0] == look_for:
                print(line_elements)
                if p_val != None:
                    raise RuntimeError('already found p value! must wrongly parsed the result')
                p_val = line_elements[5 - offset]
                if p_val.startswith('<'):
                    p_val = 0
                else:
                    try:
                        p_val = float(p_val)
                    except ValueError as e:
                        # with open(str(time.time()), 'w') as f:
                        #     f.write(summary)
                        print("can't decode, reset and proceed to look for next one")
                        p_val = None
                        continue # sometimes if unlucky the lookfor will appear at the beginning of the line with data

                t_val = float(line_elements[4 - offset])
                print('t=', t_val, 'p=', p_val)
    if p_val == None:
        print(summary)
        raise ValueError('Can\'t find fit result')
    
    return p_val, t_val

def extract_glm_stats(m, look_for):
    p_val = None
    summary = str(base.summary(m))
    lines = summary.splitlines()
    print(summary)
    
    return p_val, t_val

def contamination_check(data_df, band_name, remove_whole_epoch=True):
    filtered_df_list = []
    print('Before exclusion: ', len(data_df))
    for subject in subjects:
        if remove_whole_epoch:
            to_remove_epochs_ids = []
            df_by_subject = data_df.loc[(data_df['subject_id'] == subject)]
            for ch_name in info_for_all['ch_names']:
                df_by_subject_electrode = df_by_subject.loc[df_by_subject['electrode_name'] == ch_name]
                to_remove_epochs = df_by_subject_electrode[np.abs(scipy.stats.zscore(df_by_subject_electrode['band_power'])) > surface_exclusion_by_zscore]
                to_remove_epochs_ids += to_remove_epochs['epoch_id'].to_list()
            df_by_subject = df_by_subject[~df_by_subject['epoch_id'].isin(set(to_remove_epochs_ids))]
            filtered_df_list.append(df_by_subject)
            save_cache(to_remove_epochs_ids, containmination_reject_epoch_name_generator(band_name, subject))
        else:
            for ch_name in info_for_all['ch_names']:
                df_by_subject_electrode = data_df.loc[(data_df['subject_id'] == subject) & (data_df['electrode_name'] == ch_name)]
                df_by_subject_electrode = df_by_subject_electrode[np.abs(scipy.stats.zscore(df_by_subject_electrode['band_power'])) < surface_exclusion_by_zscore]
                filtered_df_list.append(df_by_subject_electrode)
    filtered_df = pd.concat(filtered_df_list)
    print('After exclusion: ', len(filtered_df))
    return filtered_df

def reject_by_surface_contamination_check(data_df, band_name):
    filtered_df_list = []
    print('Before exclusion: ', len(data_df))
    for subject in subjects:
        df_by_subject = data_df.loc[(data_df['subject_id'] == subject)]
        if isinstance(band_name, list):
            reject_epoch_list = []
            for single_band_name in band_name:
                reject_epoch_list += load_cache(containmination_reject_epoch_name_generator(single_band_name, subject))
        else:
            reject_epoch_list = load_cache(containmination_reject_epoch_name_generator(band_name, subject))
        df_by_subject = df_by_subject[~df_by_subject['epoch_id'].isin(set(reject_epoch_list))]
        filtered_df_list.append(df_by_subject)
    filtered_df = pd.concat(filtered_df_list)
    print('After exclusion: ', len(filtered_df))
    return filtered_df

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cl

for quantities in [ 'tonic', 'vigour_constant']:
    fig, axs = plt.subplots(nrows=1, ncols=len(bands), figsize=(34, 14))
    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cbar_ax.tick_params(labelsize=24)
    plt.colorbar(cm.ScalarMappable(norm=cl.Normalize(vmin=-4.48, vmax=4.48), cmap=plt.get_cmap('RdBu_r')), cax=cbar_ax)
    if mode == 'surface':
        electrode_t_and_p_by_band = []
        for i, (band_name, band) in enumerate(bands.items()):
            band_power = []
            for psds in spectral_power:
                new_psds, _ = filter_band_psds(psds, freqs_for_all, band)
                agg_psd = np.mean(new_psds)
                if surface_power_in_db:
                    band_power.append(10 * np.log10(agg_psd))
                else:
                    band_power.append(surface_power_transform_before_lm(agg_psd))
            data_df = pd.DataFrame({'subject_id': subject_id, 'op_cost': op_cost, 'phasic_pain': phasic_pain, 
                                    'head_movement': head_movement, 'tonic': tonic, 'vigour_constant': vigour_constant, 
                                    'epoch_id': epoch_id, 'square_root_Cv_beta_dist': np.sqrt(beta_dot_distance_times_C_v), 
                                    'beta_distance': beta_distance, 'vigour_cost': vigour_cost, 'choice_value': choice_value,
                                    'band_power': band_power, 'electrode_name': electrode_name, 'beta_dot_distance_times_C_v': beta_dot_distance_times_C_v,
                                    'square_root_vigour_constant': np.sqrt(vigour_constant), 'square_root_beta_dist': np.sqrt(beta_distance),
                                    'pressure': pressure_val, 'pressure_rating': pressure_rating_val,
                                    'pressure_tonic_only': pressure_val_tonic_only, 'pressure_rating_tonic_only': pressure_rating_val_tonic_only})
            if surface_exclusion_by_other_band:
                data_df = reject_by_surface_contamination_check(data_df, surface_exclusion_by_other_band)
            elif surface_exclusion_by_zscore:
                data_df = contamination_check(data_df, band_name, remove_whole_epoch=True)
                # data_df.to_csv(quantities + band_name + '.csv')
            
            ch_t_value = {}
            ch_p_value = {}
            for ch_name in info_for_all['ch_names']:
                analyse_pd = data_df.loc[data_df['electrode_name'] == ch_name]

                with localconverter(ro.default_converter + pandas2ri.converter):
                    r_from_pd_df = ro.conversion.py2rpy(analyse_pd)

                if surface_regression_model == 'glmm':
                    family = rstats.poisson(link='identity')
                    m = lme4.glmer('band_power ~ ' + quantities + ' + head_movement + (1 + ' + quantities + ' | subject_id)', r_from_pd_df, family=family)
                    p_val, t_val = extract_mixed_model_stats(m, quantities, model='glmer')
                elif surface_regression_model == 'lmm':
                    m = lmer_test.lmer('band_power ~ ' + quantities + ' + head_movement + (1 + ' + quantities + ' | subject_id)', r_from_pd_df)
                    p_val, t_val = extract_mixed_model_stats(m, quantities, model='lmer')
                else:
                    raise NotImplementedError('surface regression model ' + surface_regression_model + ' not implemented')
                ch_t_value[ch_name] = t_val
                ch_p_value[ch_name] = p_val

            electrode_t_and_p_by_band.append((ch_p_value, ch_t_value, info_for_all))
            p_val = [ch_p_value[ch_name] for ch_name in info_for_all['ch_names']]
            print('before corrected: ', p_val)
            corrected_p = mne.stats.fdr_correction(p_val)
            print('after_corrected: ', corrected_p)

            mne.viz.plot_topomap([ch_t_value[ch_name] for ch_name in info_for_all['ch_names']], 
                                info_for_all, vlim=(-4.48, 4.48), show=False, axes=axs[i], mask=corrected_p[0],
                                mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k',linewidth=0, markersize=12),
                                cmap=plt.get_cmap('RdBu_r'), contours=0) # df = 30, p: 0.999 - 0.001
            axs[i].set_xlabel(band_name, fontsize=24)
        save_cache(electrode_t_and_p_by_band, 'time_frequency_surface_' + quantities + '_' + surface_power_method + str(surface_sample_range[0]) + '-' + str(surface_sample_range[1]) + ' sps' + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_'))
        plt.savefig('figures/time_frequency_t_values_' + quantities + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_') + '_fdr.png')

    elif mode == 'volume' or mode == 'surface_source':
        for i, (band_name, band) in enumerate(bands.items()):
            band_power = []
            for agg_psds in spectral_power_bf:
                band_power.append(agg_psds[i])
            data_df = pd.DataFrame({'subject_id': subject_id_bf, 'op_cost': op_cost_bf, 'phasic_pain': phasic_pain_bf,
                                    'head_movement': head_movement_bf, 'tonic': tonic_bf, 'vigour_constant': vigour_constant_bf,
                                    'vigour_cost': vigour_cost_bf, 'choice_value': choice_value_bf, 'band_power': band_power,
                                    'beta_distance': beta_distance_bf, 'square_root_vigour_constant': np.sqrt(vigour_constant_bf),
                                    'square_root_beta_dist': np.sqrt(beta_distance_bf), 'square_root_Cv_beta_dist': np.sqrt(beta_dot_distance_times_C_v_bf), 
                                    'dipole_idx': dipole_idx, 'beta_dot_distance_times_C_v': beta_dot_distance_times_C_v_bf, 'epoch_id': epoch_id,
                                    'pressure': pressure_val_bf, 'pressure_rating': pressure_rating_val_bf,
                                    'pressure_tonic_only': pressure_val_tonic_only_bf, 'pressure_rating_tonic_only': pressure_rating_val_tonic_only_bf})
            save_cache(data_df, 'volume_estimate_dataframe_' + band_name + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_') 
                       + (str(minimum_norm_noise_epoch_range[0]) + '-' + str(minimum_norm_noise_epoch_range[1]) if mode == 'surface_source' else ''))
            if not volume_early_reject:
                data_df = reject_by_surface_contamination_check(data_df, band_name)
            
            dipole_t_value = []
            dipole_p_value = []
            for dipole in range(vol_estimate_container.data.shape[0]):
                analyse_pd = data_df.loc[data_df['dipole_idx'] == dipole]

                with localconverter(ro.default_converter + pandas2ri.converter):
                    r_from_pd_df = ro.conversion.py2rpy(analyse_pd)
                
                m = lmer_test.lmer('band_power ~ ' + quantities + ' + head_movement + (1 + ' + quantities + ' | subject_id)', r_from_pd_df)
                p_val, t_val = extract_mixed_model_stats(m, quantities, model='lmer')
                dipole_t_value.append(t_val)
                dipole_p_value.append(p_val)

            vol_estimate_container.data = np.array([dipole_t_value]).T

            if mode == 'surface_source':
                save_cache(vol_estimate_container, quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_surf_src_estimate' + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_') 
                       + str(minimum_norm_noise_epoch_range[0]) + '-' + str(minimum_norm_noise_epoch_range[1]) + str(volume_early_reject) + str(covariance_method) + ('_no_source_filter' if surface_source_band_pass_filter == None else '_has-source_filter'))

            if mode == 'volume':
                if beamformer_method == 'lcmv':
                    save_cache(vol_estimate_container, quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_lcmv_surf_src_estimate' + ('_separate-vigour' if SEPARATE_VIGOUR else '_single-vigour')
                               + str(covariance_method) + str(volume_early_reject) + str(lcmv_weight_norm))
                else:
                    save_cache(vol_estimate_container, quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_vol_estimate' + ('_separate-vigour' if SEPARATE_VIGOUR else '_single-vigour') + str(covariance_method))

                    lims = [1.7, 2.46, 3.39]
                    fig = vol_estimate_container.plot(mode="stat_map",
                                                clim=dict(kind="value", pos_lims=lims),
                                                src=fwd['src'], show=False)
                    if beamformer_method == 'lcmv':
                        plt.savefig('figures/bf_time_frequency_t_values_' + quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz.png')
                    elif beamformer_method == 'dics':
                        plt.savefig('figures/bf_dics_time_frequency_t_values_' + quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz.png')

    