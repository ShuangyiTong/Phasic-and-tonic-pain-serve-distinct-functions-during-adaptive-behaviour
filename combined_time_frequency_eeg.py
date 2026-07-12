# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cl
import matplotlib as mpl

from core.utils import load_cache

surface_power_method = 'welch'
surface_sample_range = (0, 250)
quantities = 'tonic'
SEPARATE_VIGOUR = False
minimum_norm_noise_epoch_range = ('adhoc', 250)
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

fig, axs = plt.subplots(nrows=4, ncols=len(bands) + 1, figsize=(43, 38))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

from PIL import Image

for i, (band_name, band) in enumerate(bands.items()):
    counter = 0
    image = Image.open('figures/' + quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_surf_src_estimate' + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_') 
                            + str(minimum_norm_noise_epoch_range[0]) + '-' + str(minimum_norm_noise_epoch_range[1]) + 'both' + 'dorsal' + '.png')
    axs[i, counter].imshow(image)
    axs[i, counter].tick_params(axis='both', bottom=False, left=False)
    axs[i, counter].set_axis_off()
    axs[i, counter].get_xaxis().set_ticks([])
    axs[i, counter].get_yaxis().set_ticks([])
    axs[i, counter].text(0.75, 0.05, "Dorsal", fontsize=48, color='white', transform=axs[i, counter].transAxes)

    counter += 1
    for hemi in ['lh', 'rh']:
        for view in ['lateral', 'medial']:
            image = Image.open('figures/' + quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_surf_src_estimate' + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_') 
                                  + str(minimum_norm_noise_epoch_range[0]) + '-' + str(minimum_norm_noise_epoch_range[1]) + hemi + view + '.png')
            axs[i, counter].imshow(image)
            axs[i, counter].tick_params(axis='both', bottom=False, left=False)
            axs[i, counter].set_axis_off()
            axs[i, counter].get_xaxis().set_ticks([])
            axs[i, counter].get_yaxis().set_ticks([])
            if counter == 1:
                axs[i, counter].text(0.02, 0.9, band_name, fontsize=36, color='white', transform=axs[i, counter].transAxes)
                axs[i, counter].text(0.25, 0.05, "Left - Lateral", fontsize=48, color='white', transform=axs[i, counter].transAxes)
            elif counter == 2:
                axs[i, counter].text(0.25, 0.05, "Left - Medial", fontsize=48, color='white', transform=axs[i, counter].transAxes)
            elif counter == 3:
                axs[i, counter].text(0.25, 0.05, "Right - Lateral", fontsize=48, color='white', transform=axs[i, counter].transAxes)
            elif counter == 4:
                axs[i, counter].text(0.25, 0.05, "Right - Medial", fontsize=48, color='white', transform=axs[i, counter].transAxes)

            counter += 1

plt.savefig('figures/PUB/surface_source_' + quantities + '.png')