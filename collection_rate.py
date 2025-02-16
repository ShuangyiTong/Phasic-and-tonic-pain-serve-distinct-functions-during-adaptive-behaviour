# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import math
import numpy as np

import core.utils
from core.utils import load_cache, save_cache
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_ID, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series

from core.individual_subject import get_series_from_control
from core.individual_subject import apply_corrections_natural_number_indexing, get_collected_item_finally_enter_baskets
from core.experiment_data import set_expt

from core.plot import barplot_annotate_brackets

pain_cond_idx = [ "NoPainNoPressure", "NoPainTonic", "LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic"]
pain_cond_index_map = { "NoPainNoPressure": "No phasic pain \n+ No tonic pain",
                        "NoPainTonic": "No phasic pain \n+ With tonic pain",
                        "LowNoPressure": "Low phasic pain \n+ No tonic pain",
                        "LowTonic": "Low phasic pain\n+ With tonic pain",
                        "HighNoPressure": "High phasic pain\n+ No tonic pain",
                        "HighTonic": "High phasic pain\n+ With tonic pain"  }
start_trial_for_analysis = 6
end_trial_for_analysis = 24
EXPT_NAME = 'Expt4'
exclude_participants = ['SUB14', 'SUB20']
set_expt(EXPT_NAME)

all_collection_rates = None #load_cache('collection rate' + EXPT_NAME)
if all_collection_rates:
    exp_data = make_experiment_data(exclude_participants=exclude_participants, exclusive_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, UNITY_DEVICE_NAME])
else:
    exp_data = make_experiment_data(exclude_participants=exclude_participants, exclusive_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, EEGLAB_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION])
subjects = list(exp_data.keys())

all_pain_conds = get_multiple_series(exp_data, lambda individual_data: list(map(
    lambda msg: msg.split('-')[-1], 
    get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[start_trial_for_analysis:end_trial_for_analysis], subjects)

if all_collection_rates == None:
    all_collection_rates = get_multiple_series(exp_data, lambda individual_data: apply_corrections_natural_number_indexing(individual_data, get_collected_item_finally_enter_baskets(individual_data), 'rate_correction_factor',
                                                            'multiply')[start_trial_for_analysis:end_trial_for_analysis], subjects)
    save_cache(all_collection_rates, 'collection rate' + EXPT_NAME)