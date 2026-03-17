import numpy as np
import pytest


def compute_fft_spectrum(signal):
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / 1000.0)
    spectrum = np.abs(np.fft.rfft(signal))
    return freqs, spectrum


def find_dominant_frequency(freqs, spectrum):
    peak_idx = np.argmax(spectrum)
    return peak_idx, freqs[peak_idx]


def classify_signal(peak_freq):
    if peak_freq < 10.0:
        return "low"
    elif peak_freq < 100.0:
        return "mid"
    return "high"


def test_fft_output_shape():
    signal = np.zeros(256)
    freqs, spectrum = compute_fft_spectrum(signal)
    assert len(freqs) == 129
    assert len(spectrum) == 129
    assert freqs.shape == spectrum.shape


def test_dominant_frequency_index():
    t = np.arange(512) / 1000.0
    signal = np.sin(2 * np.pi * 50.0 * t)
    freqs, spectrum = compute_fft_spectrum(signal)
    peak_idx, _ = find_dominant_frequency(freqs, spectrum)
    assert peak_idx == 26


def test_signal_classification_low():
    label = classify_signal(5.0)
    assert label == "low"


def test_signal_classification_mid():
    label = classify_signal(50.0)
    assert label == "mid"


def test_signal_classification_high():
    label = classify_signal(200.0)
    assert label == "high"


def test_spectrum_length_even():
    signal = np.random.default_rng(0).standard_normal(512)
    freqs, spectrum = compute_fft_spectrum(signal)
    assert len(spectrum) == len(signal) // 2 + 1


def test_channel_count():
    channels = [compute_fft_spectrum(np.zeros(128)) for _ in range(8)]
    assert len(channels) == 8


def test_argmax_returns_integer():
    spectrum = np.array([1.0, 5.0, 3.0, 2.0])
    idx = int(np.argmax(spectrum))
    assert idx == 1
    assert isinstance(idx, int)
