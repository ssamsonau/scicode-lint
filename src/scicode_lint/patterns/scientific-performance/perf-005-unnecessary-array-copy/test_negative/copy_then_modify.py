import numpy as np


def apply_bandpass_filter(signal, sample_rate, low_freq, high_freq):
    filtered = signal.copy()
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / sample_rate)
    spectrum = np.fft.rfft(filtered)
    spectrum[freqs < low_freq] = 0
    spectrum[freqs > high_freq] = 0
    filtered[:] = np.fft.irfft(spectrum, n=len(filtered))
    filtered -= filtered.mean()
    return filtered


def clip_outliers_inplace(measurements, n_sigma=3):
    cleaned = measurements.copy()
    mu = cleaned.mean()
    sigma = cleaned.std()
    cleaned[cleaned > mu + n_sigma * sigma] = mu + n_sigma * sigma
    cleaned[cleaned < mu - n_sigma * sigma] = mu - n_sigma * sigma
    return cleaned
