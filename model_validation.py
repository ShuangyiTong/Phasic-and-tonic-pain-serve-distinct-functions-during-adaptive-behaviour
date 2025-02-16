# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import sys
import os
import json
import pandas as pd
import math
import numpy as np

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME, save_cache
from core.experiment_data import set_expt
from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_lazy_subject_indexed

Expt_name = sys.argv[1]
set_expt(Expt_name)

ACCURACY_EPSILON = 1e-4
cpp_fitted_params = None

from core.individual_subject import get_fruit_position_map, get_abstract_action_v2, get_trial_start_timestamps

import realtime_model

accuracy = []

def convert_cpp_params_to_python(fitted_params, duplicate_vigour_for_phasic_pain_cond=False):
    if Expt_name == 'Expt4':
        '''C++
        enum PARAMETER
        {
            PAIN_FUNC_X_SCALE,
            PAIN_FUNC_X_TRANSLATE,
            VIGOUR_CONSTANT_NO_PAIN,
            VIGOUR_CONSTANT_LOW_PAIN,
            VIGOUR_CONSTANT_HIGH_PAIN,
            AVERAGE_REWARD,
            PAIN_FUNC_SCALE,
            HORIZONTAL_DISTANCE_COF, // vertical coefficient is normalized by horizontal
            NUM_FREE_PARAMETER
        };
        Python
        [FITTING_GRID_SIGMOID_X_SCALE, 
        FITTING_GRID_SIGMOID_X_TRANSLATION, 
        FITTING_GRID_VIGOUR ..., 
        FITTING_GRID_REWARD, 
        FITTING_GRID_PAIN] + [HORIZONTAL_DISTANCE_COF, VERTICAL_DISTANCE_COF]
        '''
        if duplicate_vigour_for_phasic_pain_cond:
            return [fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[2],
                    fitted_params[2], fitted_params[3], fitted_params[4], fitted_params[5],
                    math.sqrt(1 - fitted_params[5] ** 2)]
        return [fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[3],
                fitted_params[4], fitted_params[5], fitted_params[6], fitted_params[7],
                math.sqrt(1 - fitted_params[7] ** 2)]
    elif Expt_name == 'Expt2':
        return [fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[3],
                fitted_params[4], fitted_params[5], fitted_params[6], fitted_params[7],
                fitted_params[8], fitted_params[9], math.sqrt(1 - fitted_params[9] ** 2)]
    else:
        raise NotImplemented

def global_sim_check(fitted_params, pineapple_maps, pain_conditions, behavioural_data):
    accs = [realtime_model.in_block_simulation(convert_cpp_params_to_python(fitted_params), pineapple_maps[x], pain_conditions[x], *(behavioural_data[x]), visualization_verbose=True, block_id=x) 
            for x in range((realtime_model.end_trial_for_analysis if realtime_model.end_trial_for_analysis else 0) - realtime_model.start_trial_for_analysis)]
    if abs(np.nanmean(accs) - fitted_params[-1]) > ACCURACY_EPSILON:
        print("Model fitting check failed - CPP:", fitted_params[-1], " >>>>>>>>> Python:", np.nanmean(accs))
        return False
    else:
        print("Pass - CPP:", fitted_params[-1], " ======== Python:", np.nanmean(accs))
    accuracy.append(fitted_params[-1])
    return True
    
def check_one(individual_data, subject):
    fitted_params = cpp_fitted_params[subject]
    pineapple_maps = get_fruit_position_map(individual_data)[realtime_model.start_trial_for_analysis:realtime_model.end_trial_for_analysis]
    behavioural_data = [get_abstract_action_v2(individual_data, ts) for ts in get_trial_start_timestamps(individual_data)][realtime_model.start_trial_for_analysis:
                                                                                                                           realtime_model.end_trial_for_analysis]
    pain_conditions = realtime_model.get_pain_conditions(individual_data)
    print(len(pineapple_maps), len(behavioural_data))
    print(pain_conditions)

    return global_sim_check(fitted_params, pineapple_maps, pain_conditions, behavioural_data)

if __name__ == '__main__':
    exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME], lazy_closure=True)
    subjects = list(exp_data.keys())
    with open(sys.argv[2]) as f:
        cpp_fitted_params = json.load(f)
    res = get_multiple_series_lazy_subject_indexed(exp_data, check_one, subjects)
    print(res)

    if all(res):
        print("====== All", len(res), "tests passed ======")
    else:
        print("====== Not passed ======")

    print(np.mean(accuracy))

