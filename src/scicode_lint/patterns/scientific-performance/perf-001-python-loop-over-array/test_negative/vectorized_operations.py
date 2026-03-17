import numpy as np


def detect_outliers_zscore(measurements, threshold=3.0):
    mean = np.mean(measurements, axis=0)
    std = np.std(measurements, axis=0)
    z_scores = np.abs((measurements - mean) / std)
    return np.any(z_scores > threshold, axis=1)


def moving_average(signal, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="valid")


def spectral_energy(fft_coefficients, freq_bins, low_freq, high_freq):
    mask = (freq_bins >= low_freq) & (freq_bins <= high_freq)
    return np.sum(np.abs(fft_coefficients[mask]) ** 2)


sensor_data = np.random.randn(1000, 6)
outlier_mask = detect_outliers_zscore(sensor_data)
clean_data = sensor_data[~outlier_mask]

time_series = np.random.randn(5000)
smoothed = moving_average(time_series, window_size=50)
