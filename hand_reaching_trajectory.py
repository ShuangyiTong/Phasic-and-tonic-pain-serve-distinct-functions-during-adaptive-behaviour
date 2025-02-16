# Copyright (c) 2023 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import math
import numpy as np
import core.utils
from core.utils import load_cache, save_cache
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_ID, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_lazy

from core.individual_subject import get_series_from_control
from core.individual_subject import apply_corrections_natural_number_indexing
from core.individual_subject import get_fruit_picking_moving_distance, get_fruit_position_map, fruit_to_prev_gaze_head_distance, get_all_collected_fruits, get_collected_item_finally_enter_baskets
from core.experiment_data import set_expt
from core.individual_subject import get_pick_up_trajectory

from core.plot import barplot_annotate_brackets
EXPT_NAME = 'Expt4'
set_expt(EXPT_NAME)
CACHE_NAME = 'hand trajectory'
GLOABL_SPEED = False
reaching_trajectory = None #load_cache(CACHE_NAME + EXPT_NAME + 'AllSpeed')
if reaching_trajectory:
    exp_data = make_experiment_data(exclude_participants=['SUB20', 'SUB14'], exclusive_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, UNITY_DEVICE_NAME], lazy_closure=True)
else:
    exp_data = make_experiment_data(exclude_participants=['SUB20', 'SUB14'], exclusive_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION], lazy_closure=True)
subjects = list(exp_data.keys())

pain_cond_idx = [ "NoPain", "MidLowPain", "MidMidPain", "MidHighPain", "MaxPain" ]
start_trial_for_analysis = -10
end_trial_for_analysis = None
if EXPT_NAME == 'Expt4':
    start_trial_for_analysis = 6
    end_trial_for_analysis = 24

    pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]

if reaching_trajectory == None:
    calculate_trajectory_by_cond = lambda cond: get_multiple_series_lazy(exp_data, lambda individual_data: [x for x, c in 
                                                         zip(get_pick_up_trajectory(individual_data,
                                                                                get_fruit_position_map(individual_data),
                                                                                lambda x: True,# not x.endswith('G'),
                                                                                granularity=21)[start_trial_for_analysis:end_trial_for_analysis],
                                                             list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 
                                                                                                                              'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis])
                                                        if cond in c], subjects)
    reaching_trajectory_tonic = calculate_trajectory_by_cond('Tonic')
    reaching_trajectory_no_tonic = calculate_trajectory_by_cond('NoPressure')
    reaching_trajectory = (reaching_trajectory_tonic, reaching_trajectory_no_tonic)
    print(reaching_trajectory)
    save_cache(reaching_trajectory, CACHE_NAME + EXPT_NAME + 'AllSpeed')
else:
    reaching_trajectory_tonic = reaching_trajectory[0]
    reaching_trajectory_no_tonic = reaching_trajectory[1]