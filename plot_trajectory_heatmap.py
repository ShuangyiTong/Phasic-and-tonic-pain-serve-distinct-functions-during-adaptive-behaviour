# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import core.utils
core.utils.verbose = True

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME
from core.utils import BASKETS_LOCATION_X, BASKETS_LOCATION_Z

from core.experiment_data import set_expt

set_expt('Expt2')

pain_cond_idx = [ "NoPain", "MidLowPain", "MidMidPain", "MidHighPain", "MaxPain" ]

from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_lazy

from core.individual_subject import get_series_from_control
from core.individual_subject import get_2d_moving_trajectory
from core.utils import save_cache, load_cache

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB11"], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME], lazy_closure=True)

subjects = list(exp_data.keys())
trials_for_analysis = 10

CACHE_NAME = 'allocentric_map'
results = None # load_cache(CACHE_NAME)
if not results:
    all_trajectories = get_multiple_series_lazy(exp_data, lambda individual_data: get_2d_moving_trajectory(individual_data)[-trials_for_analysis:], subjects)
    all_pain_conds = get_multiple_series_lazy(exp_data, lambda individual_data: list(map(
        lambda msg: msg.split('-')[-1], 
        get_series_from_control(individual_data, 'log', 'msg', 'Main task session end, trial: ', 'msg')))[-trials_for_analysis:], subjects)
    save_cache((all_trajectories, all_pain_conds), CACHE_NAME)
else:
    all_trajectories = results[0]
    all_pain_conds = results[1]

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from core.utils import trajectory_distance_calculator

np.set_printoptions(threshold=sys.maxsize)
move_distance_by_cond = [[], [], [], [], []]
x_coors_by_cond = [[], [], [], [], []]
z_coors_by_cond = [[], [], [], [], []]
total_x_coors = []
total_z_coors = []
for pain_conds, coors in zip(all_pain_conds, all_trajectories):
    for pain_cond, coor in zip(pain_conds, coors):
        try:
            x_coor, z_coor= zip(*coor)
        except TypeError:
            print(coor)
            continue
        x_coors_by_cond[pain_cond_idx.index(pain_cond)] += x_coor
        z_coors_by_cond[pain_cond_idx.index(pain_cond)] += z_coor
        total_x_coors += x_coor
        total_z_coors += z_coor
        move_distance_by_cond[pain_cond_idx.index(pain_cond)].append(trajectory_distance_calculator(coor))

plt.figure(figsize=(9, 9))
total_heatmap, xedges, yedges = np.histogram2d(total_x_coors, total_z_coors, range=[[114, 129], [107, 122]], bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.scatter(BASKETS_LOCATION_X, BASKETS_LOCATION_Z, marker='o', s=200, color='red')
plt.imshow(total_heatmap.T, extent=extent, origin='lower', cmap='plasma', norm=mpl.colors.Normalize(vmin=0, vmax=500, clip=True))
plt.axis('off')
plt.savefig('figures/PUB/Expt2_global_trajectory_heat.svg')