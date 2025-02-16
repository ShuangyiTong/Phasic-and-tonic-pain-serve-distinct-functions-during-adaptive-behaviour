# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import math
import numpy as np

CONTROL_DATA_KEYWORD = '__Control_Data__'
DEVICE_DATA_KEYWORD = '__Device_Data_'
DEVICE_DESCRIPTOR_KEYWORD = '__Device_Descriptors__'
DATA_NOTES_KEYWORD = '__Data_Notes__'

NI_DEVICE_NAME = "Digitimer_DS5_NI_USB_6202"
UNITY_DEVICE_NAME = "Unity_C_Sharp_Client_VR"
ARDUINO_DEVICE_NAME = "Phys_Monitoring_Arduino_v1"
LIVEAMP_DEVICE_NAME = "LiveAmp_LSL_Compatible_Univ"
LIVEAMP_DEVICE_NAME_BRAINVISION = 'vhdr'
EEGLAB_NAME = 'markers_processed.set'

UNITY_DEVICE_ID = "o8Y6VNWF7orzDfPGCrJh"
NI_DEVICE_ID = "SC91BBkyiIWxnJMipKYk"

verbose = False
def verbose_out(msg: str):
    if verbose:
        print(msg)

BASKETS_LOCATION_X = [116.6852, 128.4352, 128.2452, 117.2952, 122.4952]
BASKETS_LOCATION_Z = [119.4957, 120.4957, 111.3357, 111.4457, 116.0757]

class a:
    '''Incorrect implementation of dependent type, terrible, but looks better
    '''
    def __init__(self) -> None:
        pass

class a1(a):
    def __init__(self) -> None:
        pass

from scipy.signal import butter, lfilter, medfilt, medfilt2d
from scipy.ndimage import median_filter as scipy_medfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

from scipy import signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def median_filter(data, window):
    return medfilt(data, window)

def remove_label_duplicated(ax):
    handles, labels = ax.get_legend_handles_labels()
    i =1
    while i<len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i +=1

    return handles, labels

def nannorm(xs):
    sum = 0
    for x in xs:
        if x == x:
            sum += x * x

    return math.sqrt(sum)

def nannorm_l1(xs):
    sum = 0
    for x in xs:
        if x == x:
            sum += x
    return sum

def nan_square_sum(xs):
    sum = 0
    for x in xs:
        if x == x:
            sum += x * x

    return sum

def trajectory_distance_calculator(raw_trajectory):
    trajectory = np.array(raw_trajectory)
    cumulative_distance = 0
    prev_point = trajectory[0]
    for point in trajectory[1:]:
        cumulative_distance += np.linalg.norm(np.array(prev_point) - np.array(point))
        prev_point = point

    return cumulative_distance / 3 # divide by 3 as we scaled by 3 in the game

import dill

def save_cache(obj, name):
    with open('temp/' + name + '.dill', 'wb') as f:
        dill.dump(obj, f)

def load_cache(name):
    try:
        with open('temp/' + name + '.dill', 'rb') as f:
            obj = dill.load(f)
        return obj
    except OSError:
        print('IO error: did mistype the file name "' + 'temp/' + name + '.dill"?')
        return None

from collections import OrderedDict
class FixSizeOrderedDict(OrderedDict):
    # Author: https://stackoverflow.com/a/49274421/7733679
    def __init__(self, *args, maxlen=0, **kwargs):
        self._max = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)

import math

def sigmoid(x, m):
    return m * ((2 * math.exp(x)) / (math.exp(x) + 1) - 1)

def gaussian_kernel(x, tau, sigma):
    return (1 / math.sqrt(2 * math.pi * sigma)) * math.exp(-((x-tau)**2) / (2 * sigma**2))

def construct_crf_sequence(events, event_start_timestamp=0, total_length_seconds=60, sfreq=10):
    # CRF function data: https://www.sciencedirect.com/science/article/pii/S0167876010000115
    tau_hat = 3.0745
    sigma_hat = 0.7013
    lambda1_hat = 0.3176
    lambda2_hat = 0.0708

    total_length = total_length_seconds * sfreq

    N = [gaussian_kernel(t / float(sfreq), tau_hat, sigma_hat) for t in range(total_length)]
    BiExp = [math.exp(-lambda1_hat * (t / float(sfreq))) + math.exp(-lambda2_hat * (t / float(sfreq)))
             for t in range(total_length)]
    
    # Use truncated N and BiExp: In practice, the Gaussian function was truncated within a centered window of 4 s and the CRF was specified for the time interval
    # of 0–120 s and was set to 0 everywhere else 
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC2772899/
    response = np.convolve(N, BiExp)[:total_length]
    # import matplotlib.pyplot as plt
    # plt.plot(response)
    # plt.show()

    output_array = np.zeros(total_length)

    for event in events:
        ts = int(((event[3] - event_start_timestamp) / 1000.0) * sfreq)
        # event trigger is just 1, adding the truncated response function directly due to LTI property
        output_array[ts:] += response[:total_length - ts]
    
    return output_array