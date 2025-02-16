# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import sys
import itertools
EPSILON = 1e-10
Expt_name = sys.argv[1]
print("running: " + Expt_name)

USE_PAIN_RATINGS = False
USE_OBJECTIVE_CURRENT_VALUES = True
# Fitting grid preset for pain ratings
FITTING_GRID_SIGMOID_X_SCALE = slice(1, 3, 1)
FITTING_GRID_SIGMOID_X_TRANSLATION = slice(-10, 10, 10)
FITTING_GRID_SIGMOID_X_TRANSLATION_SUBTICK = 1
# FITTING_GRID_DISCOUNT_FACTOR = [0, 0.2, 0.4, 0.6, 0.8, 1]
# FITTING_GRID_DISCOUNT_SUBTICK = 0.1
FITTING_GRID_VIGOUR_NOPAIN = slice(10, 30, 10)
FITTING_GRID_VIGOUR_LOWPAIN = slice(10, 30, 10)
FITTING_GRID_VIGOUR_HIGHPAIN = slice(10, 30, 10)
FITTING_GRID_REWARD = slice(1, 2, 1)
FITTING_GRID_PAIN = slice(10, 30, 10)
if USE_OBJECTIVE_CURRENT_VALUES:
    FITTING_GRID_SIGMOID_X_TRANSLATION = slice(-1, 1, 1)
    FITTING_GRID_SIGMOID_X_TRANSLATION_SUBTICK = 0.1
FIT_MODE_BY_BLOCK = False
NO_CONDITION_1 = False
NO_CONDITION_2 = True
CONDITION_1_NAME = 'Finger'
if Expt_name == 'Expt4':
    CONDITION_1_NAME = 'Tonic'
CONDITION_2_NAME = 'Back'
if Expt_name == 'Expt4':
    CONDITION_2_NAME = 'NoPressure'

import math
from collections import deque

from core.utils import FixSizeOrderedDict

from core.sim_env import eye_tracked_obs_v2

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_NAME, UNITY_DEVICE_ID, ARDUINO_DEVICE_NAME, nannorm, nan_square_sum, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION
from core.experiment_data import set_expt

set_expt(Expt_name)

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_subprocessed_lazy

from core.individual_subject import get_abstract_action_v2
from core.individual_subject import get_fruit_position_map
from core.individual_subject import get_trial_start_timestamps
from core.individual_subject import get_trial_end_timestamps
from core.individual_subject import apply_corrections_natural_number_indexing
from core.individual_subject import get_end_of_trial_pain_ratings

import numpy as np
import scipy.optimize as opt
from scipy.stats import trim_mean
from datetime import datetime
from time import mktime
import dill

from core.individual_subject import get_series_from_control
from core.plot import generate_map_visualization

import pandas as pd

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION], lazy_closure=True)

subjects = list(exp_data.keys())
if Expt_name == 'Expt3':
    start_trial_for_analysis = 5
    end_trial_for_analysis = 15
elif Expt_name == 'Expt4':
    start_trial_for_analysis = 6
    end_trial_for_analysis = 24
else:
    start_trial_for_analysis = -10
    end_trial_for_analysis = None

NUM_PARAMS = 1
PREDEFINED_PARAMS = 7
if Expt_name == 'Expt1':
    NUM_PARAMS = 2
if Expt_name == 'Expt2':
    PREDEFINED_PARAMS = 9

def combined_model(memory,
                   horizontal_dist_cost,
                   vertical_dist_cost,
                   pain_cost,
                   pain_cond,
                   extra_params,
                   average_reward_per_millisecond,
                   average_speed_m_per_millisecond,
                   current_delay_in_millisecond,
                   get_item=None,
                   detailed_dump=False):
    average_speed_m_per_millisecond *= 1000
    current_delay_in_millisecond /= 1000
    # extra param 0, 1: x scale and x translation; 2: C_v; 3: reward
    choice_values = []
    if Expt_name in ['Expt3', 'Expt4']:
        vigour_constant = extra_params[2] if pain_cond[0] < EPSILON else (extra_params[4] if 1 - pain_cond[0] < EPSILON else extra_params[3])
    else:
        vigour_constant = extra_params[2 + round(pain_cond[0] / 0.25)]
    for pineapple, coor in memory.items():
        if get_item and get_item != pineapple:
            continue

        beta_dot_distance_times_C_v = vigour_constant * (horizontal_dist_cost * math.sqrt(coor[0]**2 + coor[2]**2) + vertical_dist_cost * abs(coor[1]))

        phasic_pain = (pain_cost / (1 + math.exp(-extra_params[0] * (pain_cond[0] - extra_params[1]))) if pineapple.endswith('G') else 0) # sigmoid
        choice_value = phasic_pain
        current_tau = (current_delay_in_millisecond + (math.sqrt(coor[0]**2 + coor[2]**2 + coor[1]**2) / average_speed_m_per_millisecond))
        choice_value -= beta_dot_distance_times_C_v / current_tau
        choice_value -= average_reward_per_millisecond * current_tau

        if pineapple == get_item or get_item == '__ANY__':
            if detailed_dump:
                return { 'op_cost': average_reward_per_millisecond * current_tau,
                         'phasic_pain': phasic_pain,
                         'vigour_cost': beta_dot_distance_times_C_v / current_tau,
                         'choice_value': choice_value,
                         'vigour_constant': vigour_constant,
                         'beta_dot_distance_times_C_v': beta_dot_distance_times_C_v,
                         'beta_distance': (horizontal_dist_cost * math.sqrt(coor[0]**2 + coor[2]**2) + vertical_dist_cost * abs(coor[1])) }
            else:
                return get_item, choice_value, None

    if get_item:
        return None, None, None # no pineapple in memory or not found

np.seterr(all='raise')
def in_block_simulation(optim_params, pineapple_map, pain_condition, pickup_choices, frames, eye_tracking, handedness, c_vigor_pain=None, c_vigor_no_pain=None, dump=False, visualization_verbose=False, block_id=0, dump_key_epochs=False):
    if CONDITION_2_NAME in pain_condition[1] and NO_CONDITION_2:
        return math.nan
    if CONDITION_1_NAME in pain_condition[1] and NO_CONDITION_1:
        return math.nan

    if len(eye_tracking) < 5:
        print('Returned no enough eye tracking data')
        return math.nan
    if FIT_MODE_BY_BLOCK:
        print(optim_params)
        if nannorm(optim_params[PREDEFINED_PARAMS:]) > 1:
            return math.nan

    if Expt_name == 'Expt1':
        pain_cost = 0
        horizontal_dist_cost = optim_params[PREDEFINED_PARAMS]
        vertical_dist_cost = optim_params[PREDEFINED_PARAMS + 1]
        extra_params = optim_params[:PREDEFINED_PARAMS]
    else:
        pain_cost = optim_params[PREDEFINED_PARAMS - 1]
        horizontal_dist_cost = optim_params[PREDEFINED_PARAMS]
        vertical_dist_cost = optim_params[PREDEFINED_PARAMS + 1]

        if horizontal_dist_cost < 0 or vertical_dist_cost < 0:
            return math.nan

        extra_params = optim_params[:PREDEFINED_PARAMS - 1] # ignore pain cost

    current_map = pineapple_map.copy()
    in_pickup = False
    memory = FixSizeOrderedDict(maxlen=7)
    last_seen_pineapple_ts = {}
    print(extra_params)

    correct_frame = 0.0
    total_frame = 0.0
    skip_after_pick = 0
    eye_tracking_miss = 0

    time_error = []
    last_picked_pineapple_ts = frames[0]['timestamp']

    last_non_choice_optimal_value = -float('inf')
    last_non_optimal_choice = None
    need_reevaluation = True
    prediction_is_correct_for_now = False

    object_picked_up = ""
    pickup_choice_counter = 0
    start_time_point = frames[0]['timestamp']
    skip_current_pickup = False
    # y_position = frames[0]['head_position'][1]
    player_point = 'head_position' # placeholder, comment below to use head_position only
    if handedness == 'left':
        player_point = 'left_hand_position'
    elif handedness == 'right':
        player_point = 'right_hand_position'
    else:
        print('pain side not defined correctly in data notes')
        exit()

    distance_travelled = 0
    last_frame_coor = np.array(frames[0][player_point])
    current_avg_speed = 0
    average_reward_rate = extra_params[PREDEFINED_PARAMS - 2]

    if dump:
        dump_array = { 'timestamp': [], 'gaze_object': [], 'picked_object': [], 'object_in_basket': [], 'new_pineapple': [], 'average_speed': []}
        for i in range(7):
            dump_array['mem' + str(i)] = []
            dump_array['mem_x' + str(i)] = []
            dump_array['mem_y' + str(i)] = []
            dump_array['mem_z' + str(i)] = []
    if dump_key_epochs:
        key_epoch_dump = { 'timestamp_ms': [], 'op_cost': [], 'phasic_pain': [], 'vigour_cost': [], 'choice_value': [], 'head_movement': [], 'tonic': [], 'vigour_constant': [], 'beta_dot_distance_times_C_v': [], 'beta_distance': [] }
    print(pain_condition)
    for frame_id, frame in enumerate(frames):
        memory, new_pineapple = eye_tracked_obs_v2(frame['gaze_object'], memory, frame[player_point], pineapple_map=current_map)
        distance_travelled += np.linalg.norm(np.array(frame[player_point]) - last_frame_coor)
        last_frame_coor = np.array(frame[player_point])
        if frame['timestamp'] - start_time_point <= 0:
            current_avg_speed = 1e-3 # 20m per block bootstrap value, but we have a multiplier 3 in the unity scene
        else:
            current_avg_speed = distance_travelled / (frame['timestamp'] - start_time_point)
        if dump:
            dump_array['timestamp'].append(frame['timestamp'] - start_time_point)
            dump_array['gaze_object'].append(frame['gaze_object'])
            dump_array['picked_object'].append(frame['pickable_object_attached'])
            dump_array['object_in_basket'].append(frame['pickable_object'])
            new_pineapple = new_pineapple if new_pineapple.startswith('PA') else ""
            dump_array['new_pineapple'].append(new_pineapple)
            dump_array['average_speed'].append(current_avg_speed)
            for i, (k, d) in enumerate(memory.items()):
                dump_array['mem' + str(i)].append(k)
                dump_array['mem_x' + str(i)].append(d[0])
                dump_array['mem_y' + str(i)].append(d[1])
                dump_array['mem_z' + str(i)].append(d[2])
            for i in range(7 - len(memory)):
                dump_array['mem' + str(7-i-1)].append('')
                dump_array['mem_x' + str(7-i-1)].append(0)
                dump_array['mem_y' + str(7-i-1)].append(0)
                dump_array['mem_z' + str(7-i-1)].append(0)
        # print(memory)
        if frame['gaze_object'].startswith("PA"):
            last_seen_pineapple_ts[frame['gaze_object']] = frame['timestamp']
        if not in_pickup:
            if pickup_choice_counter < len(pickup_choices):
                if not skip_current_pickup:
                    if (frame['timestamp'] - start_time_point) > 0:
                        choice, choice_value, choice_values = combined_model(memory, horizontal_dist_cost, vertical_dist_cost, 
                                                                pain_cost, pain_condition, extra_params,
                                                                average_reward_per_millisecond=average_reward_rate,
                                                                average_speed_m_per_millisecond=current_avg_speed,
                                                                current_delay_in_millisecond=frame['timestamp'] - last_picked_pineapple_ts,
                                                                get_item=frame['gaze_object'])
                        if choice:
                            if choice == pickup_choices[pickup_choice_counter]:
                                if need_reevaluation:
                                    if dump_key_epochs and frame_id + 10 < len(frames):
                                        d_dump = combined_model(memory, horizontal_dist_cost, vertical_dist_cost, 
                                                                pain_cost, pain_condition, extra_params,
                                                                average_reward_per_millisecond=average_reward_rate,
                                                                average_speed_m_per_millisecond=current_avg_speed,
                                                                current_delay_in_millisecond=frame['timestamp'] - last_picked_pineapple_ts,
                                                                get_item=frame['gaze_object'], detailed_dump=True)
                                        print(choice_value)
                                        print(d_dump['phasic_pain'] - d_dump['op_cost'] - d_dump['vigour_cost'])
                                        assert(choice_value - (d_dump['phasic_pain'] - d_dump['op_cost'] - d_dump['vigour_cost']) < EPSILON)
                                        assert(choice_value - d_dump['choice_value'] < EPSILON)
                                        key_epoch_dump['op_cost'].append(d_dump['op_cost'])
                                        key_epoch_dump['phasic_pain'].append(d_dump['phasic_pain'])
                                        key_epoch_dump['vigour_cost'].append(d_dump['vigour_cost'])
                                        key_epoch_dump['choice_value'].append(d_dump['choice_value'])
                                        key_epoch_dump['vigour_constant'].append(d_dump['vigour_constant'])
                                        key_epoch_dump['beta_dot_distance_times_C_v'].append(d_dump['beta_dot_distance_times_C_v'])
                                        key_epoch_dump['beta_distance'].append(d_dump['beta_distance'])
                                        key_epoch_dump['timestamp_ms'].append(frame['timestamp'])
                                        key_epoch_dump['head_movement'].append((np.linalg.norm(np.array(frames[frame_id - 10]['head_position']) - np.array(frames[frame_id + 10]['head_position'])) * 3) # scale by scale factor
                                                                                / (frames[frame_id - 10]['timestamp'] - frames[frame_id + 10]['timestamp']) * 1000) # m/ms -> m/s
                                        key_epoch_dump['tonic'].append(int(not NO_CONDITION_1))
                                    if last_non_choice_optimal_value > choice_value:
                                        prediction_is_correct_for_now = False
                                        print("INCORRECT:", choice, '->', choice_value, '<', last_non_choice_optimal_value)
                                    else:
                                        prediction_is_correct_for_now = True
                                        print("CORRECT:", choice, '->', choice_value, '>', last_non_choice_optimal_value)
                                    need_reevaluation = False
                            else:
                                if last_non_choice_optimal_value < choice_value:
                                    last_non_choice_optimal_value = choice_value
                                    last_non_optimal_choice = choice
                                    print("Update non choice value:", choice, '->', choice_value)
                                need_reevaluation = True
            if frame['pickable_object_attached'] != "":
                if prediction_is_correct_for_now:
                    correct_frame += 1
                    print('==========Correct, Expected:', frame['pickable_object_attached'], '=========')
                else:
                    print('==========Incorrect, non_optimal ->', last_non_choice_optimal_value, 'Expected',
                            frame['pickable_object_attached'], '========="')
                last_non_choice_optimal_value = -float('inf')
                need_reevaluation = True
                prediction_is_correct_for_now = False

                object_picked_up = frame['pickable_object_attached']
                pickup_choice_counter += 1
                in_pickup = True

                last_picked_pineapple_ts = frame['timestamp']
                total_frame += 1
        else:
            skip_after_pick += 1
            if frame['pickable_object_attached'] == "":
                current_map[object_picked_up] = { 'x': frame[player_point][0], 'y': 0, 'z': frame[player_point][2] } # after being repicked up, we reference from this position, although true physics won't directly falls down
                object_picked_up = ""
                in_pickup = False # TODO: Change this to include final object in as final reward
                skip_current_pickup = False
                # memory.clear()

        if frame['pickable_object'] != "":
            try:
                memory.pop(frame['pickable_object'])
            except KeyError:
                eye_tracking_miss += 1
            current_map.pop(frame['pickable_object'])
            if skip_current_pickup and frame['pickable_object'] == choice:
                print("drop previous prediction because it is in the basket now: " + choice)
                choice = None
                skip_current_pickup = False
            if last_non_optimal_choice == frame['pickable_object']:
                last_non_choice_optimal_value = -float('inf')
                last_non_optimal_choice = None
                need_reevaluation = True
        
        if 'deleted_fruit' in frame and frame['deleted_fruit'] != "":
            # print("fruit deleted", frame['deleted_fruit'])
            if frame['deleted_fruit'] in memory:
                print("deleted fruit in memory")
            try:
                current_map.pop(frame['deleted_fruit'])
                memory.pop(frame['deleted_fruit'])
            except KeyError:
                pass

        to_pop = []
        for pineapple in last_seen_pineapple_ts.keys():
            if frame['timestamp'] - last_seen_pineapple_ts[pineapple] > 10000: # allow maximum 10 seconds to be considered to pick up after last seen
                try:
                    memory.pop(pineapple)
                except KeyError:
                    pass # maybe its not in memory already
                to_pop.append(pineapple)
        for pineapple in to_pop:
            last_seen_pineapple_ts.pop(pineapple)

    print(1 - correct_frame / total_frame)

    if dump:
        return pain_condition, pd.DataFrame.from_dict(dump_array)

    assert pickup_choice_counter == len(pickup_choices)

    # print("Correct frame:", correct_frame, "Total frame:", total_frame)
    # print("Skip before see:", skip_before_see, "Skip after pick:", skip_after_pick, "Eye track miss:", eye_tracking_miss)
    if c_vigor_pain is not None and c_vigor_no_pain is not None:
        acc = np.mean(np.array(time_error))
        print("time error: ", acc)
        return acc
    
    if dump_key_epochs:
        return 1 - correct_frame / total_frame, (start_time_point, key_epoch_dump)

    return 1 - correct_frame / total_frame

def global_sim(optim_params, pineapple_maps, pain_conditions, behavioural_data):
    if nannorm(optim_params[PREDEFINED_PARAMS:]) > 1:
        return 1 # return max error rate
    optim_params = [x for x in optim_params] + [math.sqrt(1 - optim_params[-1] ** 2)]
    print(optim_params)
    error_rate = [in_block_simulation(optim_params, pineapple_maps[x], pain_conditions[x], *(behavioural_data[x])) for x in range((end_trial_for_analysis if end_trial_for_analysis else 0) - start_trial_for_analysis)]
    error_rate = list(filter(lambda x: x == x, error_rate))
    if error_rate == []:
        print('Warning: empty error rate')
    print(error_rate, sum(error_rate) / len(error_rate))
    return sum(error_rate) / len(error_rate)

pain_cond_idx = [ "NoPain", "MidLowPain", "MidMidPain", "MidHighPain", "MaxPain" ]
def get_pain_cond_val(condition: str):
    if Expt_name in ["Expt1", "Expt2"]:
        return pain_cond_idx.index(condition)
    elif Expt_name == 'Expt3':
        if "High" in condition:
            return 5
        elif "Low" in condition:
            return 3
        elif "No" in condition:
            return 0
        else:
            raise Exception("Condition does not match")
    else:
        raise Exception("No such experiment condition")

def extract_stimulation_intensities_series(stimulation_intensities_ts_marked, end_timestamps):
    # iterate each end timestamp, find the last stimulation before the end_timestamp
    # it's fine if not found, because that means no pain pineapple picked up, so not affecting the computation
    ret = []
    for end_ts in end_timestamps:
        current_max_ts = 0 
        current_value = None
        for stim, marked_ts in stimulation_intensities_ts_marked:
            if marked_ts > current_max_ts and marked_ts < end_ts:
                current_max_ts = marked_ts
                current_value = float(stim.split()[-1])
        ret.append(current_value)

    return ret

def subblockwise_normalize(pain_conditions):
    def subcondition_normalize(big_block, subcondition):
        condition_max = max(list(map(lambda x: x[0], filter(lambda block: subcondition in block[1], big_block))))
        return [c if (subcondition not in c[1]) else (c[0] / condition_max ,c[1]) for c in big_block]
    if Expt_name == 'Expt3':
        big_block1 = pain_conditions[:5]
        big_block2 = pain_conditions[5:]
        big_block1 = subcondition_normalize(big_block1, 'Finger')
        big_block1 = subcondition_normalize(big_block1, 'Back')
        big_block2 = subcondition_normalize(big_block2, 'Finger')
        big_block2 = subcondition_normalize(big_block2, 'Back')
        return big_block1 + big_block2
    elif Expt_name == 'Expt4':
        big_block1 = subcondition_normalize(pain_conditions[:6], '')
        big_block2 = subcondition_normalize(pain_conditions[6:12], '')
        big_block3 = subcondition_normalize(pain_conditions[12:], '')
        return big_block1 + big_block2 + big_block3
    
def get_pain_conditions(individual_data):
    if USE_PAIN_RATINGS:
        pain_conditions = list(zip(apply_corrections_natural_number_indexing(individual_data, get_end_of_trial_pain_ratings(individual_data), 'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis],
                                   list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]))
    else:
        if USE_OBJECTIVE_CURRENT_VALUES and Expt_name not in ['Expt1', 'Expt2']:
            pain_conditions = subblockwise_normalize(list(zip(extract_stimulation_intensities_series(get_series_from_control(individual_data, 'log', 'msg', 'stimulation applied with strength', ['msg', 'timestamp']),
                                                                     get_trial_end_timestamps(individual_data)[start_trial_for_analysis:end_trial_for_analysis]),
                                   list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis])))
        else:
            pain_conditions = list(zip([get_pain_cond_val(x) / 4 for x in list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]],
                                    list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]))
    return pain_conditions

def subject_all_block_fitting(individual_data):
    pineapple_maps = get_fruit_position_map(individual_data)[start_trial_for_analysis:end_trial_for_analysis]
    behavioural_data = [get_abstract_action_v2(individual_data, ts) for ts in get_trial_start_timestamps(individual_data)][start_trial_for_analysis:end_trial_for_analysis]
    print(get_trial_start_timestamps(individual_data))
    print(len(pineapple_maps), len(behavioural_data))
    pain_conditions = get_pain_conditions(individual_data)
    print(pain_conditions)
    if FIT_MODE_BY_BLOCK:
        simulation_results_1 = [opt.brute(in_block_simulation, (slice(-1, 1, 0.1), slice(-1, 1, 0.1), slice(-1, 1, 0.1)), (pineapple_maps[x], pain_conditions[x], *(behavioural_data[x])), full_output=True, finish=None) for x in range((end_trial_for_analysis if end_trial_for_analysis else 0) - start_trial_for_analysis)]
        simulation_results_2 = [opt.brute(in_block_simulation, (slice(simulation_results_1[x][0][0] - 0.1, simulation_results_1[x][0][0] + 0.1, 0.01),
                                                    slice(simulation_results_1[x][0][1] - 0.1, simulation_results_1[x][0][1] + 0.1, 0.01), 
                                                    slice(simulation_results_1[x][0][2] - 0.1, simulation_results_1[x][0][2] + 0.1, 0.01)), (pineapple_maps[x], pain_conditions[x], *(behavioural_data[x])), full_output=True, finish=None) for x in range((end_trial_for_analysis if end_trial_for_analysis else 0) - start_trial_for_analysis)]
    else:
        if PREDEFINED_PARAMS < 1:
            simulation_results_1 = opt.brute(global_sim, [slice(-1, 1, 0.01) for _ in range(NUM_PARAMS)], (pineapple_maps, pain_conditions, behavioural_data), full_output=True, finish=None)
            simulation_results_2 = []
        else:
            simulation_results_1 = opt.brute(global_sim, [FITTING_GRID_SIGMOID_X_SCALE, FITTING_GRID_SIGMOID_X_TRANSLATION, FITTING_GRID_VIGOUR_NOPAIN, FITTING_GRID_VIGOUR_LOWPAIN, FITTING_GRID_VIGOUR_HIGHPAIN, FITTING_GRID_REWARD, FITTING_GRID_PAIN] + 
                                                          [slice(0, 0.2, 0.1) for _ in range(NUM_PARAMS)], 
                                                         (pineapple_maps, pain_conditions, behavioural_data), full_output=True, finish=None)
            simulation_results_2 = opt.brute(global_sim, [slice(simulation_results_1[0][0] - 1, simulation_results_1[0][0] + 1, 1),
                                                          slice(simulation_results_1[0][1] - FITTING_GRID_SIGMOID_X_TRANSLATION_SUBTICK, simulation_results_1[0][1] + FITTING_GRID_SIGMOID_X_TRANSLATION_SUBTICK, FITTING_GRID_SIGMOID_X_TRANSLATION_SUBTICK),
                                                          slice(simulation_results_1[0][2] - 1, simulation_results_1[0][2] + 1, 1),
                                                          slice(simulation_results_1[0][3] - 1, simulation_results_1[0][3] + 1, 1),
                                                          slice(simulation_results_1[0][4] - 1, simulation_results_1[0][4] + 1, 1),
                                                          slice(simulation_results_1[0][5], simulation_results_1[0][5] + 1, 1),
                                                          slice(simulation_results_1[0][6] - 1, simulation_results_1[0][6] + 1, 1)] + 
                                                          [slice(simulation_results_1[0][i + PREDEFINED_PARAMS], simulation_results_1[0][i + PREDEFINED_PARAMS] + 0.01, 0.01) for i in range(NUM_PARAMS)], 
                                                         (pineapple_maps, pain_conditions, behavioural_data), full_output=True, finish=None)
    print(simulation_results_1, simulation_results_2)
    t = datetime.now()
    unix_secs = str(int(mktime(t.timetuple())))
    with open('temp/res/' + unix_secs + '.dill', 'wb') as f:
        dill.dump((simulation_results_1, simulation_results_2), f)
    return (simulation_results_1, simulation_results_2)

if __name__ == "__main__":
    all_stimulation_results = get_multiple_series_subprocessed_lazy(exp_data, subject_all_block_fitting, subjects, procs=12)

    with open('temp/test' + str(NO_CONDITION_1) + str(NO_CONDITION_2) + Expt_name + '.dill', 'wb') as f:
        dill.dump([subjects, all_stimulation_results], f)