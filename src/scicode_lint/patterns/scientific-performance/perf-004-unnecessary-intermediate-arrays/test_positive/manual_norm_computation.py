import numpy as np


def apply_bandpass_filter(signal, low_freq, high_freq, sample_rate):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    mask = np.zeros_like(freqs)
    mask[(freqs >= low_freq) & (freqs <= high_freq)] = 1.0
    windowed = spectrum * mask
    filtered = np.fft.irfft(windowed, n=len(signal))
    amplitudes = np.abs(filtered)
    normalized = amplitudes / np.max(amplitudes)
    clipped = np.clip(normalized, 0.01, 0.99)
    return clipped


recording = np.random.randn(500000)
result = apply_bandpass_filter(recording, 300, 3400, 16000)
