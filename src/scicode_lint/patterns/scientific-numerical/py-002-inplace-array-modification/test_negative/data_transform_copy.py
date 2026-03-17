import numpy as np


def compute_rolling_mean(signal, window_size):
    n = len(signal)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window_size + 1)
        result[i] = signal[start : i + 1].mean()
    return result


def frequency_spectrum(time_series):
    fft_vals = np.fft.rfft(time_series)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(time_series))
    return freqs, power


def interpolate_missing(data, mask):
    indices = np.arange(len(data))
    valid = ~mask
    interpolated = np.interp(indices, indices[valid], data[valid])
    return interpolated


def pairwise_distances(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


signal = np.random.rand(200)
smoothed = compute_rolling_mean(signal, window_size=5)
freqs, power = frequency_spectrum(signal)
