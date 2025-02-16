# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cl

from core.utils import load_cache

surface_power_method = 'welch'
surface_sample_range = (0, 250)
quantities = 'vigour_constant'
SEPARATE_VIGOUR = False

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


electrode_t_and_p_by_band = load_cache('time_frequency_surface_' + quantities + '_' + surface_power_method + str(surface_sample_range[0]) + '-' + str(surface_sample_range[1]) + ' sps' + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_'))
vscale = 3.385 # 4.48

fig, axs = plt.subplots(nrows=1, ncols=len(bands), figsize=(34, 10))
cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
cbar_ax.tick_params(labelsize=32)
plt.colorbar(cm.ScalarMappable(norm=cl.Normalize(vmin=-vscale, vmax=vscale), cmap=plt.get_cmap('RdBu_r')), cax=cbar_ax)
for i, (band_name, band) in enumerate(bands.items()):
    ch_p_value, ch_t_value, info_for_all = electrode_t_and_p_by_band[i]
    p_val = [ch_p_value[ch_name] for ch_name in info_for_all['ch_names']]
    corrected_p = mne.stats.bonferroni_correction(p_val)
    uncorrected_p_mask = np.array([True if p < 0.05 else False for p in p_val])
    print("p value >>>>>")
    print(band_name, ch_p_value)
    print("t value >>>>>")
    print(band_name, ch_t_value)
    mne.viz.plot_topomap([ch_t_value[ch_name] for ch_name in info_for_all['ch_names']], 
                        info_for_all, vlim=(-vscale, vscale), show=False, axes=axs[i], mask=corrected_p[0],
                        mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k',linewidth=0, markersize=12),
                        cmap=plt.get_cmap('RdBu_r'), contours=0) # df = 30, p: 0.999 - 0.001
    mne.viz.plot_topomap([ch_t_value[ch_name] for ch_name in info_for_all['ch_names']], 
                        info_for_all, vlim=(-vscale, vscale), show=False, axes=axs[i], mask=uncorrected_p_mask,
                        mask_params=dict(marker='o', markerfacecolor='none', markeredgecolor='k',linewidth=0, markersize=12),
                        cmap=plt.get_cmap('RdBu_r'), contours=0) # df = 30, p: 0.999 - 0.001
    axs[i].set_xlabel(band_name, fontsize=48)

plt.savefig('figures/PUB/time_frequency_topo_' + quantities + '.svg')