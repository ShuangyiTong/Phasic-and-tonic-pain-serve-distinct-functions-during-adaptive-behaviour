# Copyright (c) 2023 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import numpy as np
import scipy.stats

import core.utils
core.utils.verbose = True

from core.utils import save_cache, load_cache

from core.utils import NI_DEVICE_NAME, UNITY_DEVICE_NAME, UNITY_DEVICE_ID, ARDUINO_DEVICE_NAME, LIVEAMP_DEVICE_NAME
from core.experiment_data import set_expt
from core.experiment_data import make_experiment_data
from core.experiment_data import get_multiple_series

from core.individual_subject import get_trial_end_timestamps
from core.individual_subject import get_trial_start_timestamps
from core.individual_subject import replace_event_marker_with_sample_indexed_segment_end
from core.individual_subject import event_timestamp_to_sample_indices
from core.individual_subject import get_series_from_control

from core.plot import add_significance_bar_hue, significance_converter

import mne

import time

def to_unix_timestamp(dt_obj):
    return time.mktime(dt_obj.timetuple())*1e3 + dt_obj.microsecond/1e3

EXPT_NAME = 'Expt4'
HIGH_PASS = 1 # change this to adjust high pass

set_expt(EXPT_NAME)

start_trials_for_analysis = 6
end_trials_for_analysis = 24

event_order = ["LowNoPressure", "LowTonic", "HighNoPressure", "HighTonic", "NoPainNoPressure", "NoPainTonic"]

CACHE = 'PAIN_EPOCH'
epochs_list = None #load_cache(CACHE)
if epochs_list == None:
    exp_data = make_experiment_data(exclusive_participants=[], exclude_participants=[], exclude_device_data=[NI_DEVICE_NAME, ARDUINO_DEVICE_NAME, UNITY_DEVICE_NAME, LIVEAMP_DEVICE_NAME])
    subjects = list(exp_data.keys())
    print(subjects)
    reject_criteria = dict(eeg=200e-6)
    epochs_list = get_multiple_series(exp_data, 
                lambda individual_data: 
                    (lambda raw, event_array, event_map, event_inverse_map: mne.Epochs(
                        raw,
                        np.array([[x, y, event_order.index(event_inverse_map[z])] for x, y, z in event_array if z in event_inverse_map.keys()]),
                        event_map,
                        event_repeated='drop',
                        preload=True,
                        reject=reject_criteria,
                        tmin=-0.3,
                        picks=['Cz'],
                        tmax=0.7,
                        baseline=(-0.3, 0),
                        on_missing='warn'))
                    (individual_data['eeg_clean'].set_eeg_reference('average').filter(HIGH_PASS, 30),
                    mne.events_from_annotations(individual_data['eeg_clean'])[0],
                    { k: event_order.index(k) for k, _ in mne.events_from_annotations(individual_data['eeg_clean'])[1].items() if k in event_order},
                    { v: k for k, v in mne.events_from_annotations(individual_data['eeg_clean'])[1].items() if k in event_order})
                if 'eeg_clean' in individual_data.keys() else None, subjects)

    epochs_list = list(filter(lambda x: x, epochs_list))
    save_cache(epochs_list, CACHE)

drop_log_rate = [epochs.drop_log_stats() for epochs in epochs_list]
print(drop_log_rate)
print('average drop rate:', np.mean(drop_log_rate))

if EXPT_NAME == 'Expt3':
    mne.viz.plot_compare_evokeds(dict(LowFinger=mne.grand_average([epochs["LowFinger"].average() for epochs in epochs_list]),
                                LowBack=mne.grand_average([epochs["LowBack"].average() for epochs in epochs_list]),
                                HighFinger=mne.grand_average([epochs["HighFinger"].average() for epochs in epochs_list]),
                                HighBack=mne.grand_average([epochs["HighBack"].average() for epochs in epochs_list]),
                                NoPain=mne.grand_average([epochs["NoPain"].average() for epochs in epochs_list])), axes='topo')
elif EXPT_NAME == 'Expt4':
    n1_p2_amplitudes = {}
    for group_name in event_order:
        evoked_array = [epochs[group_name].average().get_data(picks='Cz')[0] for epochs in epochs_list]
        '''https://www.sciencedirect.com/science/article/pii/S0301051123002314
        https://journals.physiology.org/doi/full/10.1152/jn.00979.2009
        search window
        N1 wave (100–170 ms)
        P2 wave (140–300 ms)
        '''
        p2_peak = [np.max(a[220:300]) for a in evoked_array] # start from epoch point
        n1_peak = [np.min(a[200:235]) for a in evoked_array] # start from epoch point
        n1_p2_amplitude = [p2 - n1 for p2, n1 in zip(p2_peak, n1_peak)]
        n1_p2_amplitudes[group_name] = n1_p2_amplitude
        print(group_name)
        print(n1_p2_amplitudes[group_name])
    
    high_high_res = scipy.stats.ttest_rel(n1_p2_amplitudes['HighTonic'], n1_p2_amplitudes['HighNoPressure'], alternative='two-sided')
    print(high_high_res)
    low_low_res = scipy.stats.ttest_rel(n1_p2_amplitudes['LowTonic'], n1_p2_amplitudes['LowNoPressure'], alternative='two-sided')
    print(low_low_res)
    high_low_tonic_res = scipy.stats.ttest_rel(n1_p2_amplitudes['HighTonic'], n1_p2_amplitudes['LowTonic'], alternative='two-sided')
    print(high_low_tonic_res)
    high_low_no_tonic_res = scipy.stats.ttest_rel(n1_p2_amplitudes['HighNoPressure'], n1_p2_amplitudes['LowNoPressure'], alternative='two-sided')
    print(high_low_no_tonic_res)

LowNoPressure = mne.grand_average([epochs["LowNoPressure"].average() for epochs in epochs_list]).get_data(picks='Cz')[0]
HighNoPressure = mne.grand_average([epochs["HighNoPressure"].average() for epochs in epochs_list]).get_data(picks='Cz')[0]
LowTonic = mne.grand_average([epochs["LowTonic"].average() for epochs in epochs_list]).get_data(picks='Cz')[0]
HighTonic = mne.grand_average([epochs["HighTonic"].average() for epochs in epochs_list]).get_data(picks='Cz')[0]

import pandas as pd

plain_analyse_pd = pd.DataFrame({ 'Amplitude': n1_p2_amplitudes['LowNoPressure'] + n1_p2_amplitudes['HighNoPressure'] + n1_p2_amplitudes['LowTonic'] + n1_p2_amplitudes['HighTonic'], 
              'Phasic': ['Low phasic pain'] * len(n1_p2_amplitudes['LowNoPressure']) + ['High phasic pain'] * len(n1_p2_amplitudes['HighNoPressure'])
               + ['Low phasic pain'] * len(n1_p2_amplitudes['LowTonic']) + ['High phasic pain'] * len(n1_p2_amplitudes['HighTonic']),
                'Tonic': ['No tonic pain'] * (len(n1_p2_amplitudes['LowNoPressure']) + len(n1_p2_amplitudes['HighNoPressure']))
                  + ['With tonic pain'] * (len(n1_p2_amplitudes['LowTonic']) + len(n1_p2_amplitudes['HighTonic'])),
                   'Subjects': [str(i) for i in range(len(n1_p2_amplitudes['LowNoPressure']))] + [str(i) for i in range(len(n1_p2_amplitudes['HighNoPressure']))]
                    + [str(i) for i in range(len(n1_p2_amplitudes['LowTonic']))] + [str(i) for i in range(len(n1_p2_amplitudes['HighTonic']))] })

import os

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

base = importr("base")
rstats = importr("stats")
rstatix = importr("rstatix")

with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(plain_analyse_pd)

fml = ro.Formula("Amplitude ~ Tonic * Phasic + Error(Subjects / (Tonic * Phasic))")
aov = rstatix.anova_test(formula=fml, data=r_from_pd_df)
print(aov)

plain_analyse_pd.to_csv('temp/rate_no_exclusion')

import os
os.system("rscript amplitude_repeated_anova" + EXPT_NAME + ".r")


import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

def erp_plot(ax, erp1, erp2, c1, c2, label1, label2, num):
    ax.plot(erp1 * 1e6, label=label1, color=c1, linewidth=3)
    ax.plot(erp2 * 1e6, label=label2, color=c2, linewidth=3)

    ax.text(-0.1, 1.1, num, transform=ax.transAxes,
        fontsize=32, fontweight='bold', va='top', ha='right')

    ax.set_xticks([50 * i for i in range(10)])
    ax.set_xticklabels(['-0.3', '-0.2', '-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6s'])
    ax.set_ylabel('\u03bcV', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    # ax.set_ylim((-10, 20)) # for 0.1, 0.5 Hz high-cut
    ax.set_ylim((-7, 12))
    ax.axvline(x=150, linestyle='dashed', c='black')

    sns.set_theme()
    sns.set_style("ticks")
    sns.despine(offset=0, trim=True, ax=ax)

    ax.legend(fontsize=18, loc='lower right')

def amplitude_plot(ax, amp1, amp2, pal, label1, label2, p_val, num):
    ax.text(-0.15, 1.15, num, transform=ax.transAxes,
        fontsize=32, fontweight='bold', va='top', ha='right')

    ort = 'v'
    analyse_pd = pd.DataFrame({'Data': np.array(amp1 + amp2) * 1e6, 'Group': [label1] * len(amp1) + [label2] * len(amp2)})
    ax=sns.stripplot( x = 'Group', y = 'Data', data = analyse_pd, palette = pal,
    edgecolor = "white",size = 6, jitter = 1, zorder = 0, alpha=1, dodge=True,
    orient = ort, ax=ax)
    ax=sns.boxplot( x = 'Group', y = 'Data', data = analyse_pd, color = "black", palette=pal,
                    width = 0.75, zorder = 10, showcaps = True, showfliers=True, ax=ax,
                    whiskerprops = {'linewidth':2, "zorder":10},
                    saturation = 1, orient = ort)
    ax.set_ylabel('', fontsize=24)
    ax.set_xlabel('', fontsize=24)
    ax.set_ylim((0, 70))
    
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))

    max_value = analyse_pd['Data'].to_numpy()
    significance_bar_vertical_len = 1
    max_value = np.nanmax(max_value) + significance_bar_vertical_len * 3
    add_significance_bar_hue(ax, 0, 1, max_value, significance_bar_vertical_len, significance_converter(p_val), p_val)

    ax.tick_params(axis='both', labelsize=18)
    sns.set_theme()
    sns.set_style("ticks")
    sns.despine(offset=0, trim=True, ax=ax)

f = plt.figure(figsize=(26, 18))

plt.subplots_adjust(hspace=0.3)

ax = f.add_subplot(3, 2, 1)

ax.set_title('No tonic pain', fontsize=24)
erp_plot(ax, LowNoPressure, HighNoPressure, sns.color_palette('Set1')[1], sns.color_palette('Set1')[0], 'Low phasic pain', 'High phasic pain', 'a')


ax = f.add_subplot(3, 2, 2)
ax.set_title('With tonic pain', fontsize=24)
erp_plot(ax, LowTonic, HighTonic, sns.color_palette('Set1')[1], sns.color_palette('Set1')[0], 'Low phasic pain', 'High phasic pain', 'b')

ax = f.add_subplot(3, 2, 3)

ax.set_title('Low phasic pain', fontsize=24)
erp_plot(ax, LowNoPressure, LowTonic, sns.color_palette('Set2')[0], sns.color_palette('Set2')[1], 'No tonic pain', 'With tonic pain', 'c')

ax = f.add_subplot(3, 2, 4)

ax.set_title('High phasic pain', fontsize=24)
erp_plot(ax, HighNoPressure, HighTonic, sns.color_palette('Set2')[0], sns.color_palette('Set2')[1], 'No tonic pain', 'With tonic pain', 'd')

ax = f.add_subplot(3, 4, 9)

ax.set_title('No tonic pain', fontsize=24)
amplitude_plot(ax, n1_p2_amplitudes['HighNoPressure'], n1_p2_amplitudes['LowNoPressure'], 'Set1', 'High\n phasic pain', 'Low\n phasic pain', high_low_no_tonic_res.pvalue, 'e')
ax.set_ylabel('N1-P2 Amplitude (\u03bcV)', fontsize=24)

ax = f.add_subplot(3, 4, 10)

ax.set_title('With tonic pain', fontsize=24)
amplitude_plot(ax, n1_p2_amplitudes['HighTonic'], n1_p2_amplitudes['LowTonic'], 'Set1', 'High\n phasic pain', 'Low\n phasic pain', high_low_tonic_res.pvalue, 'f')

ax = f.add_subplot(3, 4, 11)

ax.set_title('High phasic pain', fontsize=24)
amplitude_plot(ax, n1_p2_amplitudes['HighNoPressure'], n1_p2_amplitudes['HighTonic'], 'Set2', 'No tonic pain', 'With tonic pain', high_high_res.pvalue, 'g')

ax = f.add_subplot(3, 4, 12)

ax.set_title('Low phasic pain', fontsize=24)
amplitude_plot(ax, n1_p2_amplitudes['LowNoPressure'], n1_p2_amplitudes['LowTonic'], 'Set2', 'No tonic pain', 'With tonic pain', low_low_res.pvalue, 'h')

plt.savefig('figures/PUB/pain_erp.png')

print(np.mean(n1_p2_amplitudes['HighNoPressure']) * 1e6, np.std(n1_p2_amplitudes['HighNoPressure']) * 1e6)
print(np.mean(n1_p2_amplitudes['LowNoPressure']) * 1e6, np.std(n1_p2_amplitudes['LowNoPressure']) * 1e6)
print(np.mean(n1_p2_amplitudes['HighTonic']) * 1e6, np.std(n1_p2_amplitudes['HighTonic']) * 1e6)
print(np.mean(n1_p2_amplitudes['LowTonic']) * 1e6, np.std(n1_p2_amplitudes['LowTonic']) * 1e6)