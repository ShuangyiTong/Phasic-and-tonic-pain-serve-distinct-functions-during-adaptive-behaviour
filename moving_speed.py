# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import math
import numpy as np
import pandas as pd
import scipy.stats as stats

import core.utils
from core.utils import load_cache, save_cache
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_ID, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series

from core.individual_subject import get_series_from_control
from core.individual_subject import apply_corrections_natural_number_indexing
from core.experiment_data import set_expt
from core.individual_subject import get_3d_hand_moving_trajectory
from core.utils import trajectory_distance_calculator

import sys

EXPT_NAME = sys.argv[1]
set_expt(EXPT_NAME)
CACHE_NAME = 'moving rate'
GLOABL_SPEED = True
CACHE_EXT = 'TotalHand3dDist'
start_trial_for_analysis = 6
end_trial_for_analysis = 24
exclude_participants = ['SUB14', 'SUB20']
pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]

picking_up_speed = None #load_cache(CACHE_NAME + EXPT_NAME + CACHE_EXT)
if picking_up_speed:
    exp_data = make_experiment_data(exclude_participants=exclude_participants, exclusive_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, UNITY_DEVICE_NAME])
else:
    exp_data = make_experiment_data(exclude_participants=exclude_participants, exclusive_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION])
subjects = list(exp_data.keys())

all_pain_conds = get_multiple_series(exp_data, lambda individual_data: list(map(
    lambda msg: msg.split('-')[-1], 
    get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], subjects)

if picking_up_speed == None:
    picking_up_speed = get_multiple_series(exp_data, lambda individual_data: apply_corrections_natural_number_indexing(individual_data, [trajectory_distance_calculator(coor) / 60 for coor in get_3d_hand_moving_trajectory(individual_data)], 'rate_correction_factor', 'multiply')[start_trial_for_analysis:end_trial_for_analysis], subjects)

    save_cache(picking_up_speed, CACHE_NAME + EXPT_NAME + CACHE_EXT)