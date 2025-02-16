# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import os
import numpy as np
import math
import statistics
import time
import scipy.spatial.transform as transform

import mne

from typing import List, Tuple, Callable, Union, Any

from datasets.manage import get_datapack_names, load_datapack
from core.utils import CONTROL_DATA_KEYWORD, DEVICE_DATA_KEYWORD, DEVICE_DESCRIPTOR_KEYWORD, DATA_NOTES_KEYWORD
from core.utils import UNITY_DEVICE_ID, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME
from core.utils import verbose_out
from core.utils import butter_bandpass_filter, butter_lowpass_filter, butter_highpass_filter, median_filter, construct_crf_sequence

def brainvision_eeg_dataloader(datapack_name):
    raw = mne.io.read_raw_brainvision(datapack_name, preload=True)
    return raw

def make_individual_data(subject_name: str, exclude_device_data: List[str] = []) -> dict:
    individual_data = {}
    individual_data['name'] = subject_name
    for datapack_name in get_datapack_names(subject_name):
        if CONTROL_DATA_KEYWORD in datapack_name:
            individual_data['control_data'] = load_datapack(datapack_name)
        elif DEVICE_DESCRIPTOR_KEYWORD in datapack_name:
            individual_data['device_descriptor'] = load_datapack(datapack_name)
        elif DATA_NOTES_KEYWORD in datapack_name:
            individual_data['data_notes'] = load_datapack(datapack_name)
        elif DEVICE_DATA_KEYWORD in datapack_name:
            # remove last '__.json'
            device_name = datapack_name[datapack_name.find(DEVICE_DATA_KEYWORD) + len(DEVICE_DATA_KEYWORD):-7]
            if device_name not in exclude_device_data:
                verbose_out("Loading datapack: " + datapack_name + " for device: " + device_name)
                individual_data[device_name] = load_datapack(datapack_name)
        elif LIVEAMP_DEVICE_NAME_BRAINVISION in datapack_name and LIVEAMP_DEVICE_NAME_BRAINVISION not in exclude_device_data:
            individual_data['eeg'] = brainvision_eeg_dataloader(datapack_name)
            individual_data['eeg'].set_montage(mne.channels.make_standard_montage('brainproducts-RNP-BA-128'), on_missing='ignore')
        elif EEGLAB_NAME in datapack_name and EEGLAB_NAME not in exclude_device_data:
            individual_data['eeg_clean'] = mne.io.read_raw_eeglab(datapack_name, preload=True)
            individual_data['eeg_clean'].set_montage(mne.channels.make_standard_montage('brainproducts-RNP-BA-128'), on_missing='ignore')

    return individual_data

def swap_block(individual_data: dict, processed_block_series: List):
    if 'swap_block' in individual_data['data_notes']:
        for swap_tuple in individual_data['data_notes']['swap_block']:
            a = processed_block_series[swap_tuple[0] - 1]
            processed_block_series[swap_tuple[0] - 1] = processed_block_series[swap_tuple[1] - 1]
            processed_block_series[swap_tuple[1] - 1] = a

    return processed_block_series

def get_series_from_control(individual_data: dict, device_id: str, check_field: str, substr: str, retrieve_field: Union[str, List[str]]) -> List:
    return list(map(lambda x: x[retrieve_field] if isinstance(retrieve_field, str) else [x[r] for r in retrieve_field], 
                filter(lambda x: (x['timestamp'] < individual_data['data_notes']['control_ignore_time_range'][0] or 
                                  x['timestamp'] > individual_data['data_notes']['control_ignore_time_range'][1])
                                if 'control_ignore_time_range' in individual_data['data_notes'] else True,
                filter(lambda x: substr in x[check_field] if check_field in x else False, individual_data['control_data'][device_id]))))

def get_end_of_trial_pain_ratings(individual_data: dict) -> List[int]:
    log_prefix = 'Pain ratings: '
    return list(map(lambda x: round(float(x[len(log_prefix):]) * 10), 
                get_series_from_control(individual_data, 'log', 'msg', log_prefix, 'msg')))

def get_trial_end_timestamps(individual_data: dict) -> List[int]:
    return get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'timestamp')

def get_calibrate_ratings_submit_timestamps(individual_data: dict) -> List[int]:
    return get_series_from_control(individual_data, UNITY_DEVICE_ID, 'set_board_main_text', 'Click Start when you are ready', 'timestamp')

def get_trial_start_timestamps(individual_data: dict) -> List[int]:
    return get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'timestamp')

def get_electric_shock_timestamps(individual_data: dict) -> List[int]:
    return get_series_from_control(individual_data, 'log', 'msg', 'stimulation applied with strength', 'timestamp')

def get_electric_shock_strength(individual_data: dict) -> List[int]:
    return list(map(lambda x: float(x[len('stimulation applied with strength '):]), get_series_from_control(individual_data, 'log', 'msg', 'stimulation applied with strength', 'msg')))

def get_immediate_value_before_timestamp(data_type: str, device_name: str, timestamps: List[int], field: str, individual_data: dict) -> List:
    '''data_type (str): 'control' | 'report'
    '''
    current_timestamp_ind = 0
    values = []
    temp_val = None

    frames = None
    if data_type == 'report':
        frames = individual_data[device_name]
    elif data_type == 'control':
        frames = individual_data['control_data'][device_name]
        
    for frame in frames:
        if frame['timestamp'] < timestamps[current_timestamp_ind]:
            if field in frame.keys():
                temp_val = frame[field]
        else:
            values.append(temp_val)
            current_timestamp_ind += 1
            if current_timestamp_ind == len(timestamps):
                return values
    
    return values

def get_immediate_value_after_timestamp(data_type: str, device_name: str, timestamps: List[int], field: str, individual_data: dict, non_empty_field=None, max_ts=31536000) -> List:
    '''data_type (str): 'control' | 'report'
    field (str): 'all' | field
    '''
    current_timestamp_ind = 0
    values = []
    timestamps_sorted_indexed = sorted([(i, ts) for i, ts in enumerate(timestamps)], key=lambda x: x[1])
    temp_val = None

    frames = None
    if data_type == 'report':
        frames = individual_data[device_name]
    elif data_type == 'control':
        frames = individual_data['control_data'][device_name]
        
    for frame in frames:
        if frame['timestamp'] > timestamps_sorted_indexed[current_timestamp_ind][1]:
            if field in frame.keys() or field == 'all':
                if non_empty_field:
                    if frame[non_empty_field] == [] or frame[non_empty_field] == "":
                        continue

                if field == 'all':
                    values.append((timestamps_sorted_indexed[current_timestamp_ind][0], frame))
                else:
                    values.append((timestamps_sorted_indexed[current_timestamp_ind][0], frame[field]))

                current_timestamp_ind += 1
                if current_timestamp_ind == len(timestamps):
                    return [x[1] for x in sorted(values, key=lambda x: x[0])]

            if frame['timestamp'] - timestamps_sorted_indexed[current_timestamp_ind][1] > max_ts:
                current_timestamp_ind += 1
                if current_timestamp_ind == len(timestamps):
                    return [x[1] for x in sorted(values, key=lambda x: x[0])]
    
    return [x[1] for x in sorted(values, key=lambda x: x[0])]

def get_frames_with_criterion(individual_data, device_name, criterion: Callable[[dict], bool]) -> List[dict]:
    frames = individual_data[device_name]
    ret = []

    for frame in frames:
        if criterion(frame):
            ret.append(frame)

    return ret

def get_continuous_frame_block_with_timestamps(individual_data: dict, device_name: str, start_ts: int, end_ts: int) -> List[dict]:
    frames = individual_data[device_name]
    ret = []

    for frame in frames:
        if frame['timestamp'] > end_ts:
            return ret
        elif frame['timestamp'] > start_ts:
            ret.append(frame)

    return ret

def get_frames_with_timestamps(individual_data: dict, device_name: str, field: str, start_timestamp: int, total_time: int) -> List[Tuple[int, Any]]:
    frames = get_continuous_frame_block_with_timestamps(individual_data, device_name, start_timestamp, start_timestamp + total_time)
    return [(frame['timestamp'], frame[field]) for frame in frames]

def get_collected_item_from_total_score(total_scores_over_trials: List[float]) -> List[int]:
    ret_list = []
    for i, v in enumerate(total_scores_over_trials):
        if i == 0:
            ret_list.append(round(total_scores_over_trials[i] * 50))
        else:
            if total_scores_over_trials[i] - total_scores_over_trials[i - 1] < 0: # a restart happened
                ret_list.append(round(total_scores_over_trials[i] * 50))
            else:
                ret_list.append(round((total_scores_over_trials[i] - total_scores_over_trials[i-1]) * 50))
    return ret_list

def get_collected_item_from_collected_log(collect_timestamps, trial_end_timestamps):
    res = [0] * len(trial_end_timestamps)
    for i, ts in enumerate(trial_end_timestamps):
        for cts in collect_timestamps:
            if cts < ts and cts > ts - 60000:
                res[i] += 1

    return res 

def get_collected_item_finally_enter_baskets(individual_data):
    lists = get_all_collected_fruits(individual_data)
    return [len(l) for l in lists]

def apply_corrections_natural_number_indexing(individual_data: dict, pre_corrected: List[int], pair_list_field: str, method: str='override') -> List[int]:
    '''method: 'multiply' | 'override'
    '''
    if pair_list_field in individual_data['data_notes']:
        for N_index, correct_val in individual_data['data_notes'][pair_list_field]:
            abs_index = N_index - 1
            if method == 'multiply':
                pre_corrected[abs_index] *= correct_val
            elif method == 'override':
                try:
                    pre_corrected[abs_index] = correct_val
                except IndexError:
                    print("Warning: Failed to correct at block: " + str(N_index) + " with value: " + str(correct_val) + " Attempting override")
                    pre_corrected += [0] * (N_index - len(pre_corrected))
                    pre_corrected[abs_index] = correct_val

    return pre_corrected

def get_natural_number_indexed_value_and_fill_gap_by_zero(individual_data: dict, field_name: str='pressure', total_trials: int=24) -> List[int]:
    field_list = individual_data['data_notes'][field_name]
    ret_list = [0] * total_trials
    for idx, v in field_list:
        ret_list[idx - 1] = v
    return ret_list

def filter_sequential_packed_timestamped_data(packed_frames: List[dict], field: str, start_timestamp: int, time_length: int, ts_offset_to_unix_ms: float=None) -> List[List[int]]:
    ret_seq = []
    last_last_frame_ts = 0
    for packed_frame in packed_frames:
        if packed_frame['timestamp'] < start_timestamp - 2000:
            # TODO: replace this 500 to a configurable variable
            continue
        elif packed_frame['timestamp'] > start_timestamp + time_length + 2000:
            break
        else:
            last_frame_ts = packed_frame['frames'][-1]['timestamp']
            if ts_offset_to_unix_ms:
                if last_last_frame_ts > last_frame_ts: # device could reset, if use unix offset things can go wrong
                    return []
                else:
                    last_last_frame_ts = last_frame_ts
                ret_seq += [[frame['timestamp'] / 1000 + ts_offset_to_unix_ms, frame[field]] for frame in packed_frame['frames']]
            else:
                ret_seq += [[packed_frame['timestamp'] - round((last_frame_ts - frame['timestamp']) / 1000), frame[field]] for frame in packed_frame['frames']]

    # align with start timestamp
    return list(filter(lambda pair: pair[0] >= start_timestamp and pair[0] <= start_timestamp + time_length, ret_seq))

def get_global_sync_timestamps(packed_frames: List[dict], start_timestamp: int) -> Union[float, None]:
    time_difference = []
    last_last_ts = 0
    for packed_frame in packed_frames:
        if packed_frame['timestamp'] < start_timestamp:
            continue
        # sample 5 seconds
        elif packed_frame['timestamp'] > start_timestamp + 5000:
            break
        else:
            last_frame_ts = packed_frame['frames'][-1]['timestamp'] / 1000
            if last_last_ts > last_frame_ts:
                return None
            else:
                last_last_ts = last_frame_ts
            time_difference.append(packed_frame['timestamp'] - last_frame_ts)

    if time_difference != []:
        return round(statistics.mean(time_difference))

    return None

def get_raw_arduino_data(individual_data: dict, field: str, start_timestamps: Union[int, List[int]], time_length: int, global_offset: Union[int, None]=0) -> List[int]:
    packed_frames = individual_data[ARDUINO_DEVICE_NAME]
    if global_offset is None:
        return []
    if isinstance(start_timestamps, list):
        return [filter_sequential_packed_timestamped_data(packed_frames, field, sts, time_length, global_offset) for sts in start_timestamps]
    else:
        return filter_sequential_packed_timestamped_data(packed_frames, field, start_timestamps, time_length, global_offset)

def emg_zeroing(series: List[List[int]], zeroing_mean=1861) -> List[List[int]]:
    return [[x[0], x[1] - zeroing_mean] for x in series]

def emg_filter(series: List[List[int]]) -> List[List[int]]:
    return [[x[0], y] for x, y in zip(series, butter_bandpass_filter(np.array(series)[:,1], 10, 200, 1000.0))]

def emg_rectify(series: List[List[int]]) -> List[List[int]]:
    return [[x[0], abs(x[1])] for x in series]

def emg_integrate(series: List[List[int]], reset_window: int = 30) -> List[List[int]]:
    acc = 0
    ret = []
    for i, x in enumerate(series):
        acc += x[1]
        if i >= reset_window:
            acc -= series[i - reset_window][1]
        ret.append((x[0], acc))
    return ret

def raw_arduino_bandpass(series: List[List[int]], low_cut: float, high_cut: float) -> List[List[int]]:
    return [[x[0], y] for x, y in zip(series, butter_bandpass_filter(np.array(series)[:,1], low_cut, high_cut, 1000.0))]

def raw_arduino_lowpass(series: List[List[int]], high_cut: float) -> List[List[int]]:
    return [[x[0], y] for x, y in zip(series, butter_lowpass_filter(np.array(series)[:,1], high_cut, 1000.0))]

def raw_arduino_highpass(series: List[List[int]], low_cut: float) -> List[List[int]]:
    return [[x[0], y] for x, y in zip(series, butter_highpass_filter(np.array(series)[:,1], low_cut, 1000.0))]

import scipy.fftpack

def frequency_response(series: List, sample_spacing: float) -> Tuple[List, List]:
    xf = np.linspace(0.0, 1.0/(2.0 * sample_spacing), len(series)//2)
    yf = scipy.fftpack.fft(series)
    yf = 2.0/len(series) * np.abs(yf[:len(series)//2])
    return xf, yf

def moving_average(interval, N=1000):
    cumsum = np.cumsum(np.insert(interval, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def segment_timestamps(data: List, to_be_segmented_ts: List[int], segment_ts: List[int], segment_length: int = 60000):
    ret = []

    # previous single loop is not correct, affecting some data results (although not significantly)
    for seg_ts in segment_ts:
        current_seg = []
        for d, ts in zip(data, to_be_segmented_ts):
            if ts < seg_ts:
                continue
            elif ts > seg_ts and ts < seg_ts + segment_length:
                current_seg.append(d)
            else:
                break
        ret.append(current_seg)

    return ret

def segment_timestamps_old(data: List, to_be_segmented_ts: List[int], segment_ts: List[int]):
    ret = []
    current_seg = []
    seg_idx = 0
    for idx, ts in enumerate(to_be_segmented_ts):
        if ts > segment_ts[seg_idx]:
            ret.append(current_seg)
            seg_idx += 1
            if seg_idx >= len(segment_ts):
                ret.append(data[idx:])
                return ret
            else:
                current_seg = [data[idx]]

        else:
            current_seg.append(data[idx])

def get_fruit_position_map(individual_data: dict) -> dict:
    fruits_frames = get_immediate_value_after_timestamp('report', UNITY_DEVICE_NAME, get_trial_start_timestamps(individual_data), 'all', individual_data, 'fruits_name')
    extra_fruits_frames = get_immediate_value_after_timestamp('report', UNITY_DEVICE_NAME, [frame['timestamp'] for frame in fruits_frames], 'all', individual_data, 'fruits_name', 1000)

    fruit_map = [{name : pos for name, pos in zip(frame['fruits_name'], frame['fruits_position']) } for frame in fruits_frames]
    extra_fruit_map = [{name : pos for name, pos in zip(frame['fruits_name'], frame['fruits_position']) } for frame in extra_fruits_frames]

    if len(fruit_map) == len(extra_fruit_map):
        print('Info: using two fruit maps')
        return [{**x, **y} for x, y in zip(fruit_map, extra_fruit_map)]
    else:
        print('Info: using single fruit maps')
        return fruit_map

def gaze_object_frame(individual_data, start_timestamp, end_timestamp, object_name):
    ret_frames = []

    frames = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_timestamp, end_timestamp)
    for frame in frames:
        if frame['gaze_object'] == object_name:
            ret_frames.append(frame)

    return ret_frames

def fruit_to_prev_head_distance(individual_data, fruit_collected_frames, fruit_map, **kwargs):
    prev_head_location = None
    distances = []
    for frame in fruit_collected_frames:
        if prev_head_location == None:
            prev_head_location = frame['head_position']
        else:
            fruit_loc = fruit_map[frame['pickable_object']]
            dist = np.linalg.norm(np.array(prev_head_location) - np.array([fruit_loc['x'], fruit_loc['y'], fruit_loc['z']]))
            distances.append((frame['pickable_object'], dist))
            prev_head_location = frame['head_position']
    return distances

def fruit_to_prev_head_distance_vertical(individual_data, fruit_collected_frames, fruit_map, **kwargs):
    prev_head_location = None
    distances = []
    for frame in fruit_collected_frames:
        if prev_head_location == None:
            prev_head_location = frame['head_position']
        else:
            fruit_loc = fruit_map[frame['pickable_object']]
            dist = abs(prev_head_location[1] - fruit_loc['y'])
            distances.append((frame['pickable_object'], dist))
            prev_head_location = frame['head_position']
    return distances

def fruit_absolute_vertical_coor(individual_data, fruit_collected_frames, fruit_map, **kwargs):
    dist = []
    for frame in fruit_collected_frames:
        fruit_loc = fruit_map[frame['pickable_object']]
        dist.append((frame['pickable_object'], fruit_loc['y']))
    return dist

def first_gaze_object_frame(individual_data, start_timestamp, end_timestamp, object_name):
    ret_frames = []

    frames = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_timestamp, end_timestamp)
    for frame in frames:
        if frame['gaze_object'] == object_name:
            return frame
        
    return None

def first_pickup_object_frame(individual_data, start_timestamp, end_timestamp, object_name):
    ret_frames = []

    frames = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_timestamp, end_timestamp)
    for frame in frames:
        if frame['pickable_object_attached'] == object_name:
            return frame

    return None

def fruit_to_prev_gaze_head_distance(individual_data, fruit_collected_frames, fruit_map, latency_option=False, egocentric=False, vertical=False, horizontal=False, override_hand=False, scale_factor=3,
                                     global_first_seen=False, start_timestamp=None, add_travel_to_basket=False, ignore_first_green=False, get_speed=False):
    prev_frame_ts = None
    distances = []
    latency = []
    egocentric_coor = []
    first_green = True
    
    for frame in fruit_collected_frames:
        if prev_frame_ts == None:
            prev_frame_ts = start_timestamp
        else:
            fruit_name = frame['pickable_object']
            fruit_loc = fruit_map[fruit_name]
            first_pickup_frame = first_pickup_object_frame(individual_data, prev_frame_ts, frame['timestamp'], fruit_name)
            if first_pickup_frame == None:
                if fruit_name.endswith('G'):
                    first_green = False
                print('Warning: object picked and collected before previous one enters basket!')
                continue
            if global_first_seen:
                first_gaze_frame = first_gaze_object_frame(individual_data, start_timestamp, frame['timestamp'], fruit_name)
            else:
                first_gaze_frame = first_gaze_object_frame(individual_data, prev_frame_ts, frame['timestamp'], fruit_name)
            if first_gaze_frame == None:
                if fruit_name.endswith('G'):
                    first_green = False
                print('Warning: eye tracking data not found for ' + fruit_name)
                continue
            if override_hand:
                handedness = individual_data['data_notes']['pain_side']
                if handedness == 'left':
                    prev_head_location = first_gaze_frame['left_hand_position']
                    pick_head_location = first_pickup_frame['left_hand_position']
                    after_head_location = frame['left_hand_position']
                elif handedness == 'right':
                    prev_head_location = first_gaze_frame['right_hand_position']
                    pick_head_location = first_pickup_frame['right_hand_position']
                    after_head_location = frame['right_hand_position']
                else:
                    print('pain side not defined correctly in data notes')
                    exit()
            else:
                prev_head_location = first_gaze_frame['head_position']
                pick_head_location = first_pickup_frame['head_position']
                after_head_location = frame['head_position']
            egocentric_coor.append((frame['pickable_object'], (np.array([fruit_loc['x'], fruit_loc['y'], fruit_loc['z']]) - np.array(prev_head_location)).tolist()))
            lat = first_pickup_frame['timestamp'] - first_gaze_frame['timestamp']
            if lat <= 0:
                if not add_travel_to_basket:
                    if frame['pickable_object'].endswith('G'):
                        first_green = False
                    print('Warning: eye tracking data found but too late for ' + fruit_name)
                    continue
                else:
                    dist = 0 # don't calculate reaching
            else:     
                if vertical:
                    dist = abs(prev_head_location[1] - fruit_loc['y']) / scale_factor
                elif horizontal:
                    dist = math.sqrt((prev_head_location[0] - fruit_loc['x']) ** 2 + (prev_head_location[2] - fruit_loc['z']) ** 2) / scale_factor
                else:
                    dist = np.linalg.norm(np.array([fruit_loc['x'], fruit_loc['y'], fruit_loc['z']]) - np.array(prev_head_location)) / scale_factor

            if not (ignore_first_green and first_green and frame['pickable_object'].endswith('G')):
                if add_travel_to_basket:
                    if dist == 0:
                        dist += np.linalg.norm(np.array(after_head_location) - np.array(prev_head_location)) / scale_factor
                    else:
                        dist += np.linalg.norm(np.array(after_head_location) - np.array(pick_head_location)) / scale_factor
                    lat = frame['timestamp'] - first_gaze_frame['timestamp']
                if get_speed:
                    distances.append((frame['pickable_object'], dist / lat))
                else:
                    latency.append((frame['pickable_object'], lat))
                    distances.append((frame['pickable_object'], dist))
            else:
                if frame['pickable_object'].endswith('G'):
                    first_green = False
            prev_frame_ts = frame['timestamp']
    # print(distances, latency)
    if latency_option:
        return latency
    if egocentric:
        return egocentric_coor
    return distances

def get_fruit_picking_moving_distance(individual_data: dict,
                                      fruit_maps: dict,
                                      fruit_type: Callable[[str], bool],
                                      distance_calculator=fruit_to_prev_head_distance,
                                      use_average: bool=True,
                                      latency: bool=False,
                                      egocentric=False,
                                      vertical=False,
                                      horizontal=False,
                                      distance_from_hand=False,
                                      scale_factor=3,
                                      global_first_seen=False,
                                      get_speed=False) -> List[int]:
    fruit_collected_frames = get_frames_with_criterion(individual_data, UNITY_DEVICE_NAME, lambda unity_frame: unity_frame['pickable_object'] != "")
    # counting from start and end can be slightly different in answers
    # the start time should be the most reliable because all timestamps during the experiment count from the start time
    # trial_start_ts = [ts - 60000 for ts in get_trial_end_timestamps(individual_data)]
    trial_start_ts = get_trial_start_timestamps(individual_data)
    print(len(trial_start_ts))
    segmented_frames = segment_timestamps(fruit_collected_frames, [frame['timestamp'] for frame in fruit_collected_frames], trial_start_ts)
    # old function miscalculated sessions with disruption
    # segmented_frames_old = segment_timestamps_old(fruit_collected_frames, [frame['timestamp'] for frame in fruit_collected_frames], trial_start_ts)[-len(trial_start_ts):]
    all_raw_distances = [distance_calculator(individual_data, frames, fruit_map, latency_option=latency, 
                                                                                 egocentric=egocentric, 
                                                                                 vertical=vertical, 
                                                                                 horizontal=horizontal, 
                                                                                 override_hand=distance_from_hand,
                                                                                 scale_factor=scale_factor,
                                                                                 global_first_seen=global_first_seen,
                                                                                 start_timestamp=start_ts,
                                                                                 get_speed=get_speed) 
                            for start_ts, frames, fruit_map in zip(trial_start_ts, segmented_frames, fruit_maps)]
    filtered_raw_distances = list(map(lambda distances: list(filter(lambda pair: fruit_type(pair[0]), distances)), all_raw_distances))

    if use_average:
        return [sum(x) / len(x) if x != [] else math.nan for x in [list(map(lambda pair: pair[1], pairs)) for pairs in filtered_raw_distances]]
    else:
        return [x if x != [] else math.nan for x in [list(map(lambda pair: pair[1], pairs)) for pairs in filtered_raw_distances]]

def carrying_speed(individual_data, start_timestamp, end_timestamp, scale_factor=3):
    frames = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_timestamp, end_timestamp)
    is_carry = False
    prev_coor = []
    prev_time = None
    distance_travelled = 0
    time_spent = 0
    for frame in frames:
        handedness = individual_data['data_notes']['pain_side']
        if handedness == 'left':
            player_location = frame['left_hand_position']
        elif handedness == 'right':
            player_location = frame['right_hand_position']
        else:
            print('pain side not defined correctly in data notes')
            exit()
        if frame['pickable_object_attached'] != "":
            if is_carry:
                distance_travelled += np.linalg.norm(np.array(player_location) - np.array(prev_coor)) / scale_factor
                time_spent += frame['timestamp'] - prev_time

            prev_coor = player_location
            prev_time = frame['timestamp']
            is_carry = True
        else:
            is_carry = False

    return distance_travelled / time_spent

def get_abstract_action(individual_data, start_timestamp, time_interval=1500, max_search_time=60000):
    timestamps = []
    abstract_actions = []
    head_positions = []
    head_rotations = []
    gaze_objects = []
    has_gazed_object = False
    current_head_position = None
    current_head_rotation = None
    current_gaze_objects = []
    local_ts = 0
    already_picked_up = False
    prev_pickup = None
    decided_action = None
    disabled_interval = False
    frames = individual_data[UNITY_DEVICE_NAME]
    for frame in frames:
        if frame['timestamp'] < start_timestamp:
            continue
        
        if frame['timestamp'] > start_timestamp + max_search_time:
            return timestamps, abstract_actions, head_positions, head_rotations, (gaze_objects if has_gazed_object else False)
        
        current_gaze_objects.append(frame['gaze_object'])
        if not has_gazed_object:
            if frame['gaze_object'] != '':
                has_gazed_object = True

        if frame['timestamp'] > start_timestamp + time_interval * local_ts:
            current_head_position = frame['head_position']
            current_head_rotation = frame['head_rotation']
            if local_ts == 0:
                local_ts += 1
                continue
            
            if not disabled_interval or decided_action:
                timestamps.append(local_ts)
                abstract_actions.append(decided_action if decided_action else 'search')
                head_positions.append(current_head_position)
                head_rotations.append(current_head_rotation)
                gaze_objects.append(current_gaze_objects)
                current_gaze_objects = []
            local_ts += 1
            decided_action = None
            disabled_interval = False

        if not decided_action:
            if frame['pickable_object_attached'] != "":
                if not already_picked_up:
                    already_picked_up = True

                    # avoid short pick-ups
                    if frame['pickable_object_attached'] != prev_pickup:
                        decided_action = frame['pickable_object_attached']
                        prev_pickup = frame['pickable_object_attached']
                    else:
                        disabled_interval = True

        if already_picked_up and frame['pickable_object_attached'] == "":
            already_picked_up = False
            disabled_interval = True

    return timestamps, abstract_actions, head_positions, head_rotations, (gaze_objects if not has_gazed_object else False)

def get_2d_moving_trajectory(individual_data: dict) -> List[List[int]]:
    trial_start_ts = get_trial_start_timestamps(individual_data)
    all_frames = [get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, ts, ts + 60000) for ts in trial_start_ts]

    return [[(frame['head_position'][0], frame['head_position'][2]) for frame in frames] for frames in all_frames]

def get_3d_hand_moving_trajectory(individual_data: dict) -> List[List[int]]:
    trial_start_ts = get_trial_start_timestamps(individual_data)
    all_frames = [get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, ts, ts + 60000) for ts in trial_start_ts]

    handedness = individual_data['data_notes']['pain_side']
    frame_position = None
    if handedness == 'left':
        frame_position = 'left_hand_position'
    elif handedness == 'right':
        frame_position = 'right_hand_position'
    else:
        print('pain side not defined correctly in data notes')
        exit()

    return [[(frame[frame_position][0], frame[frame_position][1], frame[frame_position][2]) for frame in frames] for frames in all_frames]

def get_non_dominant_hand_3d_moving_trajectory_rel_head(individual_data: dict) -> List[List[Tuple[float, float, float]]]:
    handedness = individual_data['data_notes']['pain_side']
    if handedness == 'left':
        hand = 'right_hand_position'
    elif handedness == 'right':
        hand = 'left_hand_position'
    else:
        print('pain side not defined correctly in data notes')
        exit()

    trial_start_ts = get_trial_start_timestamps(individual_data)
    all_frames = [get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, ts, ts + 60000) for ts in trial_start_ts]

    rotation_vecs = [[transform.Rotation.from_quat((frame['head_rotation'][1], # Unity in w,x,y,z; scipy in x,y,z,w
                                      frame['head_rotation'][2], 
                                      frame['head_rotation'][3], 
                                      frame['head_rotation'][0])).as_rotvec() for frame in frames] for frames in all_frames]
    inverse_change_of_basis = [[np.linalg.inv(np.column_stack((np.array([vec[0], vec[2]]) / np.linalg.norm([vec[0], vec[2]]),
                                                               np.array([vec[2], -vec[0]]) / np.linalg.norm([vec[2], -vec[0]])))) for vec in rotation_vec] for rotation_vec in rotation_vecs]
    
    before_rotated = [[(frame[hand][0] - frame['head_position'][0], frame[hand][1], frame[hand][2] - frame['head_position'][2]) for frame in frames] for frames in all_frames]
    after_rotated_2d = [[inv_basis.dot([before_vec[0], before_vec[2]]) for before_vec, inv_basis in zip(before_vecs, inv_bases)] for before_vecs, inv_bases in zip(before_rotated, inverse_change_of_basis)]

    combine_y_after_rotated = [[(after_2d_vec[0], before_vec[1], after_2d_vec[1]) for before_vec, after_2d_vec in zip(before_vecs, after_2d_vecs)] 
                               for before_vecs, after_2d_vecs in zip(before_rotated, after_rotated_2d)]

    return combine_y_after_rotated

def get_pick_up_trajectory(individual_data, fruit_maps, pineapple_criterion=lambda x: True, granularity=10, scale_factor=3):
    def append_with_interpolation(to_be_appended, current_ts, current_proportion):
        if len(to_be_appended) < 1:
            return []
        supposed_length = int(current_proportion * granularity) + 1
        need_length = supposed_length - len(to_be_appended)
        if need_length <= 0:
            return to_be_appended
        initial_ts = to_be_appended[-1]
        interpolate_amount = (current_ts - initial_ts) / need_length
        for _ in range(need_length):
            to_be_appended.append(to_be_appended[-1] + interpolate_amount)
        return to_be_appended
    fruit_collected_frames = get_frames_with_criterion(individual_data, UNITY_DEVICE_NAME, lambda unity_frame: unity_frame['pickable_object'] != "")
    trial_start_ts = get_trial_start_timestamps(individual_data)
    print(len(trial_start_ts))
    segmented_frames = segment_timestamps(fruit_collected_frames, [frame['timestamp'] for frame in fruit_collected_frames], trial_start_ts)
    assert(len(segmented_frames) == len(trial_start_ts) == len(fruit_maps))
    handedness = individual_data['data_notes']['pain_side']
    if handedness == 'left':
        player_location = 'left_hand_position'
    elif handedness == 'right':
        player_location = 'right_hand_position'
    else:
        print('pain side not defined correctly in data notes')
        exit()
    ret_trajectories = []
    for start_ts, segmented_frame_block, fruit_map in zip(trial_start_ts, segmented_frames, fruit_maps):
        frames = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_ts, start_ts + 60000)
        current_max_distance = None
        trajectory_blockwise = []
        current_trajectory = []
        pickable_frame_counter = 0
        wait_for_in_basket = False
        for frame in frames:
            if wait_for_in_basket:
                if segmented_frame_block[pickable_frame_counter]['pickable_object'] == frame['pickable_object']:
                    wait_for_in_basket = False
            elif segmented_frame_block[pickable_frame_counter]['pickable_object'] == frame['pickable_object_attached']:
                current_trajectory = append_with_interpolation(current_trajectory, frame['timestamp'], 1.0)
                if len(current_trajectory) > 0:
                    assert(len(current_trajectory) == granularity + 1)
                    if pineapple_criterion(segmented_frame_block[pickable_frame_counter]['pickable_object']):
                        trajectory_blockwise.append([(current_max_distance / granularity) / (c - c_prev) for c, c_prev in zip(current_trajectory[1:], current_trajectory[:-1])])
                    current_trajectory = []
                current_max_distance = None
                pickable_frame_counter += 1
                if pickable_frame_counter >= len(segmented_frame_block):
                    break
            elif segmented_frame_block[pickable_frame_counter]['pickable_object'] == frame['gaze_object']:
                fruit_name = segmented_frame_block[pickable_frame_counter]['pickable_object']
                fruit_loc = fruit_map[fruit_name]
                distance = np.linalg.norm(np.array(frame[player_location]) - np.array([fruit_loc['x'], fruit_loc['y'], fruit_loc['z']])) / scale_factor
                if current_max_distance == None:
                    current_max_distance = distance
                    current_trajectory.append(frame['timestamp'])
                else:
                    current_trajectory = append_with_interpolation(current_trajectory, frame['timestamp'], 1 - distance / current_max_distance)
        if trajectory_blockwise == []:
            ret_trajectories.append(np.full(granularity, np.nan))
        else:
            ret_trajectories.append(np.mean(np.array(trajectory_blockwise), axis=0))

    return ret_trajectories

def get_labelled_pick_up_action_timestamp(individual_data: dict, block_labels: List[str], block_start_timestamps: List[int], other_action: Union[None, str]=None, remove_green_before_pickup=False) -> List[Tuple[str, bool, Union[int, None], int]]:
    ret = []
    for label, start_timestamp in zip(block_labels, block_start_timestamps):
        green_has_picked_up = False
        global_sync = get_global_sync_timestamps(individual_data[ARDUINO_DEVICE_NAME], start_timestamp)
        block_frame = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_timestamp, start_timestamp + 60000)
        already_picked_up = False
        if other_action == 'first_seen':
            already_seen = []
            for frame in block_frame:
                if frame['gaze_object'] != "" and frame['gaze_object'].startswith('PA'):
                    if frame['gaze_object'] not in already_seen:
                        already_seen.append(frame['gaze_object'])
                        isPainful = frame['gaze_object'].endswith('G')
                        if ((not remove_green_before_pickup) or green_has_picked_up) and not frame['pickable_object_attached'].endswith('G'): # avoid strong overlap with shock evoked scr
                            ret.append((label, isPainful, global_sync, frame['timestamp']))
                    if frame['pickable_object_attached'] != "" and frame['pickable_object_attached'].endswith('G'):
                        green_has_picked_up = True
        else:
            for frame in block_frame:
                if frame['pickable_object_attached'] != "":
                    if not already_picked_up:
                        already_picked_up = True
                        isPainful = frame['pickable_object_attached'].endswith('G')
                        ret.append((label, isPainful, global_sync, frame['timestamp']))

                if already_picked_up and frame['pickable_object_attached'] == "":
                    already_picked_up = False

    return ret

def resample(signal, resample_length: Tuple[int, int, int]) -> List[int]:
    start_ts = resample_length[0]
    bin_size = resample_length[1]
    new_len = resample_length[2]

    signal = np.array(signal)
    
    resampled_signal = []
    original_signal_pointer = 0

    for i in range(new_len):
        samples = []
        if original_signal_pointer >= signal.shape[0]:
            resampled_signal.append(math.nan)
            continue
        while signal[original_signal_pointer][0] < start_ts + bin_size * (i + 1):
            samples.append(signal[original_signal_pointer][1])
            original_signal_pointer += 1
            if original_signal_pointer >= signal.shape[0]:
                break
        if len(samples) == 0:
            print(signal.shape)
        resampled_signal.append(sum(samples) / len(samples) if len(samples) > 0 else math.nan)
    
    print(resampled_signal)
    return resampled_signal

def reset_to_baseline(baseline_window, signal):
    if baseline_window == 'average':
        baseline = np.nanmean(signal)
    elif baseline_window == 'min':
        baseline = np.nanmin(signal)
    elif baseline_window == 'min-1': # to fit Gamma GLM
        baseline = np.nanmin(signal) - 0.1
    else:
        baseline = signal[baseline_window]
    if baseline == math.nan:
        return None
    return [val - baseline for val in signal]

    
def grove_gsr_12_bit_adc_processing(gsr, baseline_window: int=0, required_length: int=0, resample_length: Union[Tuple[int, int, int], None]=None, apply_low_pass: Union[Tuple[float, float], None]=None):
    if gsr == []:
        return None
    analog_val = np.clip(np.array(gsr)[:,1], None, 2044) / 4 # convert from 12 bit ADC recording to 10 bit value, 1892 -> 500kOhm cutoff 2micro siemens, 1986 -> 1MOhm, 2044 -> 20MOhm
    # if np.count_nonzero(analog_val == 2044) / len(analog_val) > 0.05:
    #     return None
    resistance = ((analog_val * 2 + 1024) * 10) / (512 - analog_val) # resistance is in kOhm: https://wiki.seeedstudio.com/Grove-GSR_Sensor/ (Use Wayback machine, they've changed it)
    if len(resistance) < required_length:
        return None
    # resistance = median_filter(resistance, 1001)
    gsr_in_uS = 1000 / resistance  # convert to microsiemens: https://en.wikipedia.org/wiki/Siemens_(unit)

    for i, gsr_val in enumerate(gsr_in_uS):
        gsr[i] = [gsr[i][0], gsr_val]
    if resample_length:
        resampled_signal = resample(np.array(gsr), resample_length)
        if apply_low_pass:
            resampled_signal = butter_lowpass_filter(resampled_signal, apply_low_pass[0], apply_low_pass[1])
        resampled_signal = median_filter(resampled_signal, 5)
        # reset to baseline
        return reset_to_baseline(baseline_window, resampled_signal)
    if required_length > 0:
        return resistance[:required_length]
    return resistance

def get_gsr_segments_with_crfs(individual_data, start_timestamps,
                               block_labels, segment_per_block=None,
                               segment_length=5000, block_length=60000,
                               min_ms_passed_per_uS_change=300, resample_interval=100,
                               baseline_window='min'):
    true_gsrs = []
    seen_green_crfs = []
    pick_green_crfs = []

    seen_yellow_crfs = []
    pick_yellow_crfs = []

    for start_timestamp, block_label in zip(start_timestamps, block_labels):
        first_seen_events = get_labelled_pick_up_action_timestamp(individual_data, [block_label], [start_timestamp], 'first_seen', True)
        first_pick_up_events = get_labelled_pick_up_action_timestamp(individual_data, [block_label], [start_timestamp])

        first_seen_green_events = [e for e in first_seen_events if e[1]]
        first_pick_up_green_events = [e for e in first_pick_up_events if e[1]]
        print('>>>>>>>first_seen_green_events: ', first_seen_green_events)
        print('>>>>>>>first_pick_up_green_events: ', first_pick_up_green_events)

        first_seen_yellow_events = [e for e in first_seen_events if not e[1]]
        first_pick_up_yellow_events = [e for e in first_pick_up_events if not e[1]]

        first_seen_green_sequence = construct_crf_sequence(first_seen_green_events, event_start_timestamp=start_timestamp)
        first_pick_up_green_sequence = construct_crf_sequence(first_pick_up_green_events, event_start_timestamp=start_timestamp)

        first_seen_yellow_sequence = construct_crf_sequence(first_seen_yellow_events, event_start_timestamp=start_timestamp)
        first_pick_up_yellow_sequence = construct_crf_sequence(first_pick_up_yellow_events, event_start_timestamp=start_timestamp)

        true_gsr = []
        green_seen_crf = []
        green_pick_crf = []
        yellow_seen_crf = []
        yellow_pick_crf = []
        
        if segment_per_block:
            segment_length = int(block_length / float(segment_per_block))
            events = [current_ts * segment_length for current_ts in range(segment_per_block)]
        else:
            events = first_pick_up_events

        global_sync_time = get_global_sync_timestamps(individual_data[ARDUINO_DEVICE_NAME], start_timestamp)
        for event in events:
            if segment_per_block:
                start_crf_ts = int((event / 1000) * 10)
                end_crf_ts = start_crf_ts + int((segment_length / 1000) * 10)
                true_start_time = start_timestamp + event
            else:
                if start_timestamp + block_length - event[3] < segment_length:
                    continue
                start_crf_ts = int(((event[3] - start_timestamp) / resample_interval))
                end_crf_ts = start_crf_ts + math.ceil((segment_length / resample_interval))
                # align with integral resample_interval points. it may not be the true event time, but it matches the CRF time
                true_start_time = start_timestamp + (start_crf_ts * resample_interval)

            green_seen_crf_bin = reset_to_baseline(baseline_window, 
                                                   first_seen_green_sequence[start_crf_ts:end_crf_ts])
            green_pick_crf_bin = reset_to_baseline(baseline_window, 
                                                   first_pick_up_green_sequence[start_crf_ts:end_crf_ts])

            yellow_seen_crf_bin = reset_to_baseline(baseline_window,
                                                    first_seen_yellow_sequence[start_crf_ts:end_crf_ts])
            yellow_pick_crf_bin = reset_to_baseline(baseline_window, 
                                                    first_pick_up_yellow_sequence[start_crf_ts:end_crf_ts])
            
            gsr_raw_data = get_raw_arduino_data(individual_data, 
                                                'skin_conductance',
                                                true_start_time,
                                                segment_length,
                                                global_sync_time)
            processed_gsr = grove_gsr_12_bit_adc_processing(gsr_raw_data,
                                                            baseline_window=baseline_window,
                                                            required_length=int(segment_length * 0.8),
                                                            resample_length=(true_start_time, resample_interval, math.ceil(segment_length / resample_interval)),
                                                            apply_low_pass=None)#(4.99, 1000 / resample_interval))

            if processed_gsr != None:
                if not min_ms_passed_per_uS_change or (np.nanmax(processed_gsr) - np.nanmin(processed_gsr)) < (segment_length / min_ms_passed_per_uS_change):
                    assert (len(processed_gsr) == len(green_seen_crf_bin) == len(green_pick_crf_bin) == len(yellow_seen_crf_bin) == len(yellow_pick_crf_bin))
                    
                    true_gsr += processed_gsr
                    green_seen_crf += green_seen_crf_bin
                    green_pick_crf += green_pick_crf_bin
                    yellow_seen_crf += yellow_seen_crf_bin
                    yellow_pick_crf += yellow_pick_crf_bin

        true_gsrs.append(true_gsr)
        seen_green_crfs.append(green_seen_crf)
        pick_green_crfs.append(green_pick_crf)
        seen_yellow_crfs.append(yellow_seen_crf)
        pick_yellow_crfs.append(yellow_pick_crf)
        
    return (true_gsrs, seen_green_crfs, pick_green_crfs, seen_yellow_crfs, pick_yellow_crfs)

def dfrobot_emg_processing(emg: List[List[int]], required_length: int=0, resample_length: Tuple[int, int, int]=None) -> List[int]:
    if emg == []:
        return None
    if np.mean(np.array(emg), axis=0)[1] < 1000:
        return None
    emg = np.array(emg_integrate(emg_rectify(emg_filter(emg_zeroing(emg))), 10))
    if len(emg) < required_length or np.mean(emg, axis=0)[1] > 900000:
        return None
    if resample_length:
        return resample(emg, resample_length)
    elif required_length > 0:
        return emg[:required_length, 1]
    else:
        return emg[:required_length, 1]
    
def myowarev1_emg_processing(emg: List[List[int]], required_length: int=0, zeroing_window=500, normalise_start=450) -> List[int]:
    if emg == []:
        return None
    if len(emg) < required_length:
        return None
    if np.mean(np.array(emg), axis=0)[1] < 1000: # already integrated
        emg = np.array(emg_integrate(emg_rectify(emg_filter(emg)), 10))
    else:
        emg = np.array(emg_integrate(emg_rectify(emg_filter(emg_zeroing(emg, zeroing_mean=2048))), 10))

    ret = emg[:required_length, 1] / np.mean(emg[normalise_start:required_length, 1])
    baseline = ret[zeroing_window]
    return [ret_val - baseline for ret_val in ret]
    

def get_all_collected_fruits(individual_data: dict) -> List[List[str]]:
    start_tss = get_trial_start_timestamps(individual_data)
    ret = []
    for start_ts in start_tss:
        block_ret = []
        frames = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_ts, start_ts + 60000)
        for frame in frames:
            if frame['pickable_object'] != '':
                block_ret.append(frame['pickable_object'])
        ret.append(block_ret)

    return ret

def get_abstract_action_v2(individual_data, start_timestamp, max_distance=5, fruit_map=None):
    frames = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_timestamp, start_timestamp + 60000)
    pickup_choices = []

    eye_tracked_object = set()
    already_pick_up = False

    player_point = 'head_position'
    handedness = individual_data['data_notes']['pain_side']
    if handedness == 'left':
        player_point = 'left_hand_position'
    elif handedness == 'right':
        player_point = 'right_hand_position'
    else:
        print('pain side not defined correctly in data notes')
        exit()

    for frame in frames:
        if already_pick_up:
            if frame['pickable_object_attached'] == "":
                already_pick_up = False
        else:
            if frame['pickable_object_attached'] != "":
                already_pick_up = True
                pickup_choices.append(frame['pickable_object_attached'])
        if frame['gaze_object'] != "":
            if not frame['gaze_object'].startswith('PA'):
                continue
            if fruit_map: 
                coor = fruit_map[frame['gaze_object']]
                if math.sqrt((coor['x'] - frame[player_point][0])**2 + 
                             (coor['y'] - frame[player_point][1])**2 + 
                             (coor['z'] - frame[player_point][2])**2) / 3 > max_distance:
                    continue
            eye_tracked_object.add(frame['gaze_object'])

    return pickup_choices, frames, eye_tracked_object, handedness

def hand_speed_after_gaze_to_collected(individual_data, fruit_collected_frames, fruit_maps, fruit_filter, start_timestamps):
    speeds = []
    for start_timestamp, block_fruit_collect_frames, fruit_map in zip(start_timestamps, fruit_collected_frames, fruit_maps):
        block_speed = []
        prev_frame_ts = None
        for collect_frame in block_fruit_collect_frames:
            gaze_frame = None     
            if prev_frame_ts == None:
                prev_frame_ts = start_timestamp
            else:
                fruit_name = collect_frame['pickable_object']
                fruit_loc = fruit_map[fruit_name]
                for frame in get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, prev_frame_ts, collect_frame['timestamp']):
                    if frame['gaze_object'] == fruit_name:
                        gaze_frame = frame
                        break
                if gaze_frame == None:
                    print('warning: eye tracking object not found')
                    continue
                
                handedness = individual_data['data_notes']['pain_side']
                if handedness == 'left':
                    prev_hand_location = gaze_frame['left_hand_position']
                elif handedness == 'right':
                    prev_hand_location = gaze_frame['right_hand_position']
                else:
                    print('pain side not defined correctly in data notes')
                    exit()

                if fruit_filter(fruit_name):
                    speed = (np.linalg.norm(np.array(prev_hand_location) - np.array([fruit_loc['x'], fruit_loc['y'], fruit_loc['z']]))) / (collect_frame['timestamp'] - gaze_frame['timestamp'])
                    block_speed.append(speed)
                prev_frame_ts = collect_frame['timestamp']
        speeds.append(np.nanmean(block_speed))
    return speeds

def extract_eeg_channel_with_start_ts(individual_data, channel_name, start_timestamp, duration=60000):
    frames = get_continuous_frame_block_with_timestamps(individual_data, LIVEAMP_DEVICE_NAME, start_timestamp, start_timestamp + duration)
    return np.array([frame[channel_name] for frame in frames]).flatten()

def to_unix_timestamp(dt_obj):
    return time.mktime(dt_obj.timetuple())*1e3 + dt_obj.microsecond/1e3

def check_timestamp(t1, t2, tolerance):
    return abs(t1 - t2) < tolerance

def check_event_timestamp_against_control_data(events, origin, event_timestamps, target_event_id):
    origin_timestamp = to_unix_timestamp(origin)
    # find the first event within nms of the trigger
    timestamp_count = 0
    event_array = events[0]
    for samples, _, event_id in event_array[0]:
        if event_id == target_event_id:
            inferenced_timestamp = samples * 2 + origin_timestamp
            if timestamp_count == 0:
                # if check_timestamp(timestamp_count, inferenced_timestamp, )
                pass

def event_timestamp_to_sample_indices(origin, event_timestamps, clock_offset=0, sample_rate=500):
    origin_timestamp = int(to_unix_timestamp(origin))
    if isinstance(event_timestamps, list):
        return [(event_timestamp - origin_timestamp + clock_offset) * (sample_rate / 1000) for event_timestamp in event_timestamps]
    else:
        return (event_timestamps - origin_timestamp + clock_offset) * (sample_rate / 1000)

def replace_event_marker_with_sample_indexed_segment_end(events, indexed_segments, target_marker, replace_marker_per_segments,
                                                         start_id_num=100, remove_additional_markers=False,):
    
    event_num = events[0].shape[0]
    segment_counter = 0
    # print(indexed_segments)
    # add new marker type
    if remove_additional_markers:
        for k in list(events[1].keys()):
            if k != target_marker:
                events[1].pop(k)
    for i, replace_marker in enumerate(sorted(list(set(replace_marker_per_segments)))):
        events[1][replace_marker] = start_id_num + i
    
    # sort indexed segments and replace markers together by indexed segment's start time
    paired_sorted = sorted(zip(indexed_segments, replace_marker_per_segments), key=lambda x: x[0][0])
    indexed_segments = list(zip(*paired_sorted))[0]
    replace_marker_per_segments = list(zip(*paired_sorted))[1]

    for current_event in range(event_num):
        if events[0][current_event, 0] < indexed_segments[segment_counter][0]:
            continue
        # check the marker time matches our target modified marker
        if events[0][current_event, 2] != events[1][target_marker]:
            continue
        # go to next segment if samples count greater than current segment
        while events[0][current_event, 0] > indexed_segments[segment_counter][1]:
            segment_counter += 1
            if segment_counter >= len(indexed_segments):
                return events
        
        # check again after coming out of the loop which may mutate segment_counter
        if events[0][current_event, 0] < indexed_segments[segment_counter][0]:
            continue

        # modify markers
        events[0][current_event, 2] = events[1][replace_marker_per_segments[segment_counter]]

    return events

def add_events_to_mne_events_structure(old_event, samples, event_name, event_id):
    new_array = [[sample, 0, event_id] for sample in samples]
    old_event[1][event_name] = event_id
    
    return (np.concatenate((old_event[0], np.array(new_array))), old_event[1])

def get_gazes_by_next_pickup_and_not(individual_data, start_timestamps, flatten=True):
    all_critical_gazes = []
    all_other_gazes = []
    for start_timestamp in start_timestamps:
        frames = get_continuous_frame_block_with_timestamps(individual_data, UNITY_DEVICE_NAME, start_timestamp, start_timestamp + 60000)
        current_choice = ""
        critical_gazes = []
        other_gazes = []
        critical_map = {}
        non_critical_map = {}
        for frame in reversed(frames):
            if frame['pickable_object_attached'].startswith('PA') and current_choice != frame['pickable_object_attached']:
                current_choice = frame['pickable_object_attached']
                critical_gazes += [t for _, t in critical_map.items()] 
                other_gazes += [t for _, t in non_critical_map.items()]
                critical_map = {}
                non_critical_map = {}
            if frame['gaze_object'].startswith('PA'):
                if frame['gaze_object'] == current_choice:
                    critical_map[current_choice] = frame['timestamp']
                else:
                    non_critical_map[frame['gaze_object']] = frame['timestamp']
        if flatten:
            all_critical_gazes += critical_gazes
            all_other_gazes += other_gazes
        else:
            all_critical_gazes.append(critical_gazes)
            all_other_gazes.append(other_gazes)

    return all_critical_gazes, all_other_gazes

def check_eeg_boundary_location(event_array, event_map):
    event_inverse_map = { v: k for k, v in event_map.items()}
    trial_start_ts = None
    ts_diff = []
    for e in event_array:
        if event_inverse_map[e[2]] == 'trial_start':
            trial_start_ts = e[0]
        elif event_inverse_map[e[2]] == 'trial_end':
            ts_diff.append((e[0] - trial_start_ts) * 2)
    print(ts_diff)
    return ts_diff

def generate_eeg_forward_model(raw, volumetric_source=True, check_alignment=False, alignment_info=None):
    from mne.datasets import eegbci, fetch_fsaverage

    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)

    # The files live in:
    subject = "fsaverage"
    trans = "fsaverage"  # MNE has a built-in fsaverage transformation
    bem = fs_dir + "/bem/" + "fsaverage-5120-5120-5120-bem-sol.fif"
    mri = fs_dir + "/mri/" + "T1.mgz"

    if volumetric_source:
        # Surface fits inside a sphere with radius   98.3 mm
        # Surface contains a sphere with radius   47.7 mm
        vol_src = mne.setup_volume_source_space(
            subject, mri=mri, pos=10, bem=bem,
            add_interpolator=True, exclude=48.3,
            verbose=True)

        fwd = mne.make_forward_solution(
            raw.info, trans=trans, src=vol_src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
        )
    else:
        # src_precalculated = fs_dir + "/bem/" + "fsaverage-ico-5-src.fif"
        src_precalculated = 'temp/fsaverage-ico-3-src.fif'
        if os.path.exists(src_precalculated):
            src = src_precalculated
        else:
            src = mne.setup_source_space(subject, spacing='ico3')
            src.save(src_precalculated)
        fwd = mne.make_forward_solution(
            raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
        )
    print(fwd)

    if check_alignment:
        fig = mne.viz.plot_alignment(alignment_info, 
                                     trans=trans,
                                     subject=subject,
                                     eeg='original',
                                     surfaces=["head-dense", "white"],
                                     show_axes=True,
                                     dig=True,
                                     fwd=fwd,
                                     )
        mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))
        input('press enter to continue')

    return fwd

def new_block_start_marker_name(cond_name, block_id=None):
    if block_id:
        return cond_name + "_Start_Block_" + str(block_id)
    else:
        return cond_name + "_Start"

def add_new_start_events(event_map, event_order):
    for e in event_order:
        event_map[e] = 1400 + event_order.index(e)
    return event_map

def replace_trial_start_with_event(event_array, event_map, events, event_order, repetitive_blocks, enable_relative=False):
    event_map = add_new_start_events(event_map, event_order)
    event_inverse_map = { v: k for k, v in event_map.items()}

    to_be_updated = None
    to_be_updated_ts = None
    new_event_array = []
    condition_count = { e: 0 for e in events }

    for i, e in enumerate(event_array):
        if event_inverse_map[e[2]] in events or event_inverse_map[e[2]][:-9] in events:
            event_key = event_inverse_map[e[2]]
            if event_key.endswith('yellow_in'):
                event_key = event_key[:-9]
            if to_be_updated:
                if enable_relative:
                    new_event_array[to_be_updated] = [to_be_updated_ts, 0, 1400 + event_order.index(new_block_start_marker_name(event_key, condition_count[event_key]))]
                    condition_count[event_key] += 1
                else:
                    new_event_array[to_be_updated] = [to_be_updated_ts, 0, 1400 + event_order.index(new_block_start_marker_name(event_key))]
                to_be_updated = None
        elif event_inverse_map[e[2]] == 'trial_start':
            to_be_updated = i
            to_be_updated_ts = e[0]
        new_event_array.append(e)

    if enable_relative:
        for k, v in condition_count.items():
            print(k, v)
            assert(v == repetitive_blocks)
    
    return (np.array(new_event_array, dtype=np.int32), event_map)

def add_new_duplicate_marker_event(event_map, trial_start_markers, suffix):
    for e in trial_start_markers:
        event_map[e + suffix] = 2000 + trial_start_markers.index(e)
    return event_map

'''Must compose after replace_trial_start_with_event
Warning: Can only be called once due to add_new_duplicate_marker_event adds a hard-coded 2000
'''
def duplicate_marker_with_preceding_trial_start_name(event_array, event_map, markers_to_be_duplicate, event_order, add_suffix='_Bin'):
    trial_start_markers = event_order
    event_map = add_new_duplicate_marker_event(event_map, trial_start_markers, add_suffix)
    event_inverse_map = { v: k for k, v in event_map.items()}

    new_event_array = []
    current_trial_name = None

    for i, e in enumerate(event_array):
        new_event_array.append(e)
        if event_inverse_map[e[2]] in trial_start_markers:
            current_trial_name = event_inverse_map[e[2]]
        elif event_inverse_map[e[2]] in markers_to_be_duplicate and current_trial_name != None:
            new_event_array.append([e[0], e[1], 2000 + trial_start_markers.index(current_trial_name)])
        elif event_inverse_map[e[2]] in ['trial_start', 'trial_end']:
            current_trial_name = None
        
    return (np.array(new_event_array, dtype=np.int32), event_map)

def filter_band_psds(psds, freqs, band):
    # adapt from mne.time_frequecy.psd:205 and mne.viz.topomap.py:2812
    freq_mask = (band[0] <= freqs) & (freqs < band[1])
    freq_sl = slice(*(np.where(freq_mask)[0][[0, -1]] + [0, 1]))
    
    psds_mask = (band[0] <= freqs) & (freqs < band[1])

    filtered_psds = psds[..., psds_mask]
    filtered_freqs = freqs[freq_sl]

    assert(filtered_freqs.shape[0] == filtered_psds.shape[-1])

    return filtered_psds, filtered_freqs

def psd_calculator(e: mne.Epochs, n_fft=1000, n_overlap=500):
    ret = e.compute_psd(fmin=1, fmax=120, remove_dc=True, method='welch', n_fft=n_fft, n_overlap=n_overlap).get_data(picks='all', return_freqs=True)
    return ret

def psd_calculator_np(x):
    ret = mne.time_frequency.psd_array_welch(x=x, sfreq=500, fmin=1, fmax=100, n_fft=1000, n_overlap=500, remove_dc=True)
    return ret