import multiprocessing as mp

import numpy as np
from scipy import signal


def bandpass_filter(args):
    chunk, fs, low, high = args
    sos = signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfilt(sos, chunk)


def parallel_filter_channels(eeg_data, fs=256, low=0.5, high=40.0, n_workers=6):
    channels = [(eeg_data[ch], fs, low, high) for ch in range(eeg_data.shape[0])]
    filtered = []
    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(bandpass_filter, channels):
            filtered.append(result)
    return np.array(filtered)


recording = np.random.randn(64, 2560)
filtered_eeg = parallel_filter_channels(recording)
