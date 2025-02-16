# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import sys
EPSILON = 0.0001
Expt_name = sys.argv[1]
print("running: " + Expt_name)

import math
import random
random.seed(0)

from core.utils import FixSizeOrderedDict
from core.sim_env import eye_tracked_obs_v2

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, nannorm, nan_square_sum
from core.experiment_data import set_expt

set_expt(Expt_name)

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_subprocessed_lazy

from core.individual_subject import get_abstract_action_v2
from core.individual_subject import get_fruit_position_map
from core.individual_subject import get_trial_start_timestamps
from core.individual_subject import apply_corrections_natural_number_indexing
from core.individual_subject import get_end_of_trial_pain_ratings

import numpy as np
import scipy.optimize as opt
from datetime import datetime
from time import mktime
import dill

from core.individual_subject import get_series_from_control

t = datetime.now()
unix_secs = str(int(mktime(t.timetuple())))
log_file = open("temp/fit_log" + unix_secs + '.log', 'w')

USE_PAIN_RATINGS = False

PAIN_FUNC_PARAMS = 3

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB11"], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION], lazy_closure=True)

subjects = list(exp_data.keys())
start_trial_for_analysis = -10
end_trial_for_analysis = None

def choice_model(memory,
                 horizontal_dist_cost,
                 vertical_dist_cost,
                 pain_cost,
                 pain_cond,
                 extra_params):
    choice_values = []
    for pineapple, coor in memory.items():
        choice_value = 0
        choice_value += horizontal_dist_cost * math.sqrt(coor[0]**2 + coor[2]**2)
        choice_value += vertical_dist_cost * abs(coor[1])

        if Expt_name == 'Expt1':
            choice_value += 0 # pain_cost is set to 0, nothing interesting
        else:
            choice_value += (pain_cost / (1 + math.exp(-extra_params[0] * (pain_cond[0] - extra_params[1]))) if pineapple.endswith('G') else 0)

        choice_values.append((pineapple, choice_value))
    
    # shuffle to avoid the last in which is last seen pineapple gets picked all the time
    # because usually last pineapple is the actual choice, this gives degenerate solutions
    choice_values = random.sample(choice_values, len(choice_values))
    sorted_choices = sorted(choice_values, key=lambda x: x[1])
    if len(sorted_choices) < 1:
        return "", 0
    return sorted_choices[-1]

def get_pain_cost(optim_params):
    positivity_pain = optim_params[PAIN_FUNC_PARAMS - 1]
    # beta is first normalized by l^2 norm because geometrically, beta \cdot d(p) = \norm(beta) \norm(d(p)) \cos(\theta),
    # normalize beta by l^2 makes beta \cdot d(p) only has to do with \norm(d(p)) the euclidean distance and their angle
    # However, pain cost is colinear with beta \cdot d(p), so we just do a l^1 norm or a convex combination 1 - \norm(beta) - reward scale to improve fitting efficiency
    pain_cost = positivity_pain * (1 - math.sqrt(nan_square_sum([optim_params[PAIN_FUNC_PARAMS], optim_params[PAIN_FUNC_PARAMS + 1]])))

    return pain_cost

np.seterr(all='raise')
def in_block_simulation(optim_params, pineapple_map, pain_condition, pickup_choices, frames, eye_tracking, handedness, local_dump=False):
    if len(eye_tracking) < 5:
        log_file.write("Not enough eye tracking data found: " + str(eye_tracking))
        return math.nan
    
    if local_dump:
        dump = []

    if handedness == 'left':
        player_point = 'left_hand_position'
    elif handedness == 'right':
        player_point = 'right_hand_position'
    else:
        print('pain side not defined correctly in data notes')
        exit()

    if Expt_name == 'Expt1':
        pain_cost = 0
        horizontal_dist_cost = optim_params[PAIN_FUNC_PARAMS]
        vertical_dist_cost = optim_params[PAIN_FUNC_PARAMS + 1]
        pain_func_params_to_model = optim_params[:PAIN_FUNC_PARAMS]
    else:
        pain_cost = get_pain_cost(optim_params)
        horizontal_dist_cost = optim_params[PAIN_FUNC_PARAMS]
        vertical_dist_cost = optim_params[PAIN_FUNC_PARAMS + 1]
        pain_func_params_to_model = optim_params[:PAIN_FUNC_PARAMS]

    current_map = pineapple_map.copy()
    in_pickup = False
    memory = FixSizeOrderedDict(maxlen=7)

    correct_frame = 0.0
    total_frame = 0.0
    skip_before_see = 0
    skip_after_pick = 0
    eye_tracking_miss = 0

    object_picked_up = ""
    pickup_choice_counter = 0
    skip_current_pickup = False
    for frame in frames:
        if not in_pickup:
            if not skip_current_pickup:
                memory, new_pineapple = eye_tracked_obs_v2(frame['gaze_object'], memory, frame[player_point], pineapple_map=current_map)
                if pickup_choice_counter < len(pickup_choices) and pickup_choices[pickup_choice_counter] in memory and new_pineapple != "":
                    choice, _ = choice_model(memory, horizontal_dist_cost, vertical_dist_cost, pain_cost, pain_condition, pain_func_params_to_model)
                    dump.append({'target': pickup_choices[pickup_choice_counter], 'memory': memory.copy()})

                    # "and abs(nannorm(optim_params[PAIN_FUNC_PARAMS:])) > EPSILON " NO LONGER NEEDED WITH SHUFFLE: 
                    # if optim_params[PAIN_FUNC_PARAMS:] is 0, then newest item is chosen, then embarrassingly this is more optimal than computing the distance carefully
                    if pickup_choices[pickup_choice_counter] == choice:
                        correct_frame += 1
                    # else:
                    #     print("predicted: ", choice, " actual: ", pickup_choices[pickup_choice_counter])
                    total_frame += 1
                    skip_current_pickup = True
                else:
                    skip_before_see += 1

            if frame['pickable_object_attached']:
                object_picked_up = frame['pickable_object_attached']
                pickup_choice_counter += 1
                in_pickup = True
        else:
            skip_after_pick += 1
            if frame['pickable_object_attached'] == "":
                current_map[object_picked_up] = { 'x': frame['right_hand_position'][0], 'y': 0, 'z': frame['right_hand_position'][2] } # assume directly dropped to the ground
                object_picked_up = ""
                in_pickup = False
                skip_current_pickup = False
                memory.clear()

        if frame['pickable_object'] != "":
            try:
                memory.pop(frame['pickable_object'])
            except KeyError:
                eye_tracking_miss += 1
            current_map.pop(frame['pickable_object'])

    assert pickup_choice_counter == len(pickup_choices)

    if local_dump:
        return dump

    # print("Correct frame:", correct_frame, "Total frame:", total_frame)
    # print("Skip before see:", skip_before_see, "Skip after pick:", skip_after_pick, "Eye track miss:", eye_tracking_miss)
    return 1 - correct_frame / total_frame

def fast_in_block_simulation(optim_params, pain_condition, dump):
    if dump != dump:
        return math.nan
    
    correct_count = 0
    for trial in dump:
        pain_cost = get_pain_cost(optim_params)
        horizontal_dist_cost = optim_params[PAIN_FUNC_PARAMS]
        vertical_dist_cost = optim_params[PAIN_FUNC_PARAMS + 1]
        pain_func_params_to_model = optim_params[:PAIN_FUNC_PARAMS]

        choice, _ = choice_model(trial['memory'], horizontal_dist_cost, vertical_dist_cost, pain_cost, pain_condition, pain_func_params_to_model)
        if choice == trial['target']:
            correct_count += 1
    return 1 - correct_count / len(dump)

def global_sim(optim_params, pain_conditions, dumps):
    if nannorm(optim_params[PAIN_FUNC_PARAMS:]) > 1:
        return 1 # return max error rate

    # print(optim_params)
    error_rate = [fast_in_block_simulation(optim_params, pain_conditions[x], dumps[x]) for x in range((end_trial_for_analysis if end_trial_for_analysis else 0) - start_trial_for_analysis)]
    error_rate = list(filter(lambda x: x == x, error_rate))
    assert(error_rate != [])
    # print(error_rate, sum(error_rate) / len(error_rate))
    return sum(error_rate) / len(error_rate)

pain_cond_idx = [ "NoPain", "MidLowPain", "MidMidPain", "MidHighPain", "MaxPain" ]
def get_pain_cond_val(condition: str):
    if Expt_name in ["Expt1", "Expt2"]:
        return pain_cond_idx.index(condition) / 4.0
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

def subject_all_block_fitting(individual_data):
    pineapple_maps = get_fruit_position_map(individual_data)[start_trial_for_analysis:end_trial_for_analysis]
    behavioural_data = [get_abstract_action_v2(individual_data, ts) for ts in get_trial_start_timestamps(individual_data)][start_trial_for_analysis:end_trial_for_analysis]
    print(get_trial_start_timestamps(individual_data))
    print(len(pineapple_maps), len(behavioural_data))
    if USE_PAIN_RATINGS:
        pain_conditions = list(zip(apply_corrections_natural_number_indexing(individual_data, get_end_of_trial_pain_ratings(individual_data), 'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis],
                                   list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]))
    else:
        pain_conditions = list(zip([get_pain_cond_val(x) for x in list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]],
                                list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]))

    in_block_simulation_dumps = [in_block_simulation([1, 0, -1, 0.5, 0.5], pineapple_maps[x], pain_conditions[x], *(behavioural_data[x]), local_dump=True) 
                                    for x in range((end_trial_for_analysis if end_trial_for_analysis else 0) - start_trial_for_analysis)]

    simulation_results1 = opt.brute(global_sim, (slice(1, 10.1, 1), slice(-10, 10.1, 0.5), slice(-1, 1.1, 2), slice(-1, 1.01, 0.1), slice(-1, 1.01, 0.1)), (pain_conditions, in_block_simulation_dumps), full_output=True, finish=None)
    simulation_results2 = opt.brute(global_sim, (slice(simulation_results1[0][0] - 0.9, simulation_results1[0][0] + 0.91, 0.1), 
                                                 slice(simulation_results1[0][1] - 0.5, simulation_results1[0][1] + 0.51, 0.01),
                                                 slice(simulation_results1[0][2], simulation_results1[0][2] + 0.1, 1),
                                                 slice(simulation_results1[0][3] - 0.1, simulation_results1[0][3] + 0.11, 0.01),
                                                 slice(simulation_results1[0][4] - 0.1, simulation_results1[0][4] + 0.11, 0.01)), (pain_conditions, in_block_simulation_dumps), full_output=True, finish=None)

    print(simulation_results1, simulation_results2)
    with open('temp/res/' + unix_secs + '.dill', 'wb') as f:
        dill.dump((simulation_results1, simulation_results2), f)
    return (simulation_results1, simulation_results2)

if __name__ == "__main__":
    all_stimulation_results = get_multiple_series_subprocessed_lazy(exp_data, subject_all_block_fitting, subjects, procs=32)

    with open('temp/' + Expt_name + '_fit_fin.dill', 'wb') as f:
        dill.dump([subjects, all_stimulation_results], f)