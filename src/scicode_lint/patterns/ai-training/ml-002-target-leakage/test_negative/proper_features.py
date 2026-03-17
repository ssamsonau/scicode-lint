import numpy as np


def compute_spectral_features(signal_array, sample_rate=256):
    freqs = np.fft.rfftfreq(signal_array.shape[1], d=1.0 / sample_rate)
    power_spectrum = np.abs(np.fft.rfft(signal_array, axis=1)) ** 2

    alpha_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)

    alpha_power = power_spectrum[:, alpha_mask].mean(axis=1)
    beta_power = power_spectrum[:, beta_mask].mean(axis=1)
    total_power = power_spectrum.sum(axis=1)

    features = np.column_stack(
        [
            alpha_power / (total_power + 1e-10),
            beta_power / (total_power + 1e-10),
            np.log1p(total_power),
        ]
    )
    return features


def extract_statistical_features(windows):
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)
    skewness = np.mean(((windows - means[:, None]) / (stds[:, None] + 1e-10)) ** 3, axis=1)
    return np.column_stack([means, stds, skewness])
