# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

'''
Dump behavioural fitting data for easier empirical analysis or speeding up calculation using more native code (like C++ OpenMP or CUDA)
'''
import sys
import os
import json
import pandas as pd
import math

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME, save_cache
from core.experiment_data import set_expt
from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_lazy

Expt_name = sys.argv[1]
set_expt(Expt_name)

from core.individual_subject import get_fruit_position_map, get_abstract_action_v2, get_trial_start_timestamps

import realtime_model

def global_sim_dump_intermediate_data(pineapple_maps, pain_conditions, behavioural_data):
    # DUMMY_OPTIM_PARAMS = [0.1] * realtime_model.PREDEFINED_PARAMS + [0.1] * realtime_model.NUM_PARAMS
    if Expt_name == 'Expt4':
        DUMMY_OPTIM_PARAMS = [8, -0.4, 1, 1, 1, 0.1, -0.9, 0.1, 0.1]
    else:
        DUMMY_OPTIM_PARAMS = [8, -0.4, 1, 1, 1, 1, 1, 0.1, -0.9, 0.1, 0.1]
    return [realtime_model.in_block_simulation(DUMMY_OPTIM_PARAMS, pineapple_maps[x], pain_conditions[x], *(behavioural_data[x]), dump=True) 
            for x in range((realtime_model.end_trial_for_analysis if realtime_model.end_trial_for_analysis else 0) - realtime_model.start_trial_for_analysis)]
    # return realtime_model.in_block_simulation(DUMMY_OPTIM_PARAMS, pineapple_maps[0], pain_conditions[0], *(behavioural_data[0]), dump=True)

def dump_behavioural_data(individual_data):
    pineapple_maps = get_fruit_position_map(individual_data)[realtime_model.start_trial_for_analysis:realtime_model.end_trial_for_analysis]
    behavioural_data = [get_abstract_action_v2(individual_data, ts) for ts in get_trial_start_timestamps(individual_data)][realtime_model.start_trial_for_analysis:
                                                                                                                           realtime_model.end_trial_for_analysis]
    pain_conditions = realtime_model.get_pain_conditions(individual_data)
    print(len(pineapple_maps), len(behavioural_data))
    print(pain_conditions)

    return [x for x in global_sim_dump_intermediate_data(pineapple_maps, pain_conditions, behavioural_data) if not x is math.nan]

def deserialize(to_dumps, subjects):
    path = 'temp/behavioural_dump_json_' + Expt_name
    os.makedirs(path, exist_ok=True)
    for subject, to_dump in zip(subjects, to_dumps):
        for i, block_data in enumerate(to_dump):
            print(block_data[1].to_csv(os.path.join(path, subject + '_' + str(block_data[0][0]) + '_' + block_data[0][1] + '_' + str(i) + '.csv')))
    

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB14", "SUB20"], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME], lazy_closure=True)
subjects = list(exp_data.keys())

res = get_multiple_series_lazy(exp_data, dump_behavioural_data, subjects)
save_cache(res, "behavioural_dump_" + Expt_name)
deserialize(res, subjects)