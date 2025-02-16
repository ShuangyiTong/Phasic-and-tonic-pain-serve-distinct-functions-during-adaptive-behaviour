# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import numpy as np
import mne
Brain = mne.viz.get_brain_class()

from core.utils import NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, EEGLAB_NAME, UNITY_DEVICE_NAME, save_cache, load_cache
from core.experiment_data import set_expt
from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series_lazy_subject_indexed, get_multiple_series_lazy

from core.individual_subject import filter_band_psds, generate_eeg_forward_model, get_trial_start_timestamps, get_series_from_control

set_expt('Expt4')

exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=["SUB14", "SUB20"], 
                                exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME, LIVEAMP_DEVICE_NAME_BRAINVISION, UNITY_DEVICE_NAME], lazy_closure=True)
subjects = list(exp_data.keys())

fwd = generate_eeg_forward_model(exp_data[subjects[0]](subjects[0])['eeg_clean'], False)

subjects_dir = mne.datasets.sample.data_path() / "subjects"
print(subjects_dir)
mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir, verbose=True)

SEPARATE_VIGOUR = False
EARLY_REJECT = True
LCMV_BEAMFORMER = False
minimum_norm_noise_epoch_range = ('adhoc', 250)
covariance_method = 'empirical'
surface_source_band_pass_filter = None
lcmv_weight_norm = None
DELTA = (1, 4)
THETA = (4, 8)
ALPHA = (8, 13)
BETA = (13, 30)
GAMMA = (30, 101)
bands = {#'Delta (1-4 Hz)': DELTA,
         'Theta (4-8 Hz)': THETA,
         'Alpha (8-12 Hz)': ALPHA, 
         'Beta (12-30 Hz)': BETA,
         'Gamma (30-100 Hz)': GAMMA
}

mne.viz.set_3d_backend('pyvistaqt')

QUANTITIES_MAP = {
    'tonic': "Tonic pain",
    'vigour_constant': "Vigour Constant"
    #'square_root_vigour_constant': "Square Root Vigour Constant"
}
for quantities, quantities_name in QUANTITIES_MAP.items():
    for i, (band_name, band) in enumerate(bands.items()):
        if LCMV_BEAMFORMER:
            vol_estimate: mne.SourceEstimate = load_cache(quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_lcmv_surf_src_estimate' + ('_separate-vigour' if SEPARATE_VIGOUR else '_single-vigour')
                               + str(covariance_method) + str(EARLY_REJECT) + str(lcmv_weight_norm))
        else:
            vol_estimate: mne.SourceEstimate = load_cache(quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_surf_src_estimate' + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_') 
                        + str(minimum_norm_noise_epoch_range[0]) + '-' + str(minimum_norm_noise_epoch_range[1]) + str(EARLY_REJECT) + str(covariance_method) + ('_no_source_filter' if surface_source_band_pass_filter == None else '_has-source_filter'))

        print(fwd["nsource"], vol_estimate.data.shape[0])
        assert(fwd["nsource"] == vol_estimate.data.shape[0])

        print(sorted(vol_estimate.data.tolist()))

        for hemi in ['lh', 'rh']:
            for view in ['lateral', 'medial']:
                brains = vol_estimate.plot(hemi=hemi, smoothing_steps=30,
                                           clim={'kind': 'value', 'pos_lims': (1.310, 1.697, 3.385)}, 
                                           #clim={'kind': 'percent', 'pos_lims': (95, 97.5, 100)}, 
                                           colorbar=False)
                brains.add_annotation("aparc")
                brains.show_view(view=view)
                brains.save_image(filename='figures/' + quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_surf_src_estimate' + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_') 
                                  + str(minimum_norm_noise_epoch_range[0]) + '-' + str(minimum_norm_noise_epoch_range[1]) + hemi + view + '.png')

        brains = vol_estimate.plot(hemi='both', smoothing_steps=30,
                                    clim={'kind': 'value', 'pos_lims': (1.310, 1.697, 3.385)}, 
                                    #clim={'kind': 'percent', 'pos_lims': (95, 97.5, 100)}, 
                                    #colorbar=False
                                    )
        brains.add_annotation("aparc")
        brains.show_view(view='dorsal')
        brains.save_image(filename='figures/' + quantities + str(band[0]) + ' Hz - ' + str(band[1]) + ' Hz_surf_src_estimate' + ('_separate-vigour_' if SEPARATE_VIGOUR else '_single-vigour_') 
                            + str(minimum_norm_noise_epoch_range[0]) + '-' + str(minimum_norm_noise_epoch_range[1]) + 'both' + 'dorsal' + '.png')

print("Press any key to exit")
input()
