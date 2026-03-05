import numpy as np


def preprocess_data(data):
    result = data.copy()
    result -= result.mean()
    result /= result.std() + 1e-8
    return result


def apply_threshold(image):
    result = image.copy()
    result[result < 0.5] = 0
    result[result >= 0.5] = 1
    return result


def smooth_signal(signal):
    smoothed = signal.copy()
    for i in range(1, len(smoothed) - 1):
        smoothed[i] = (signal[i - 1] + signal[i] + signal[i + 1]) / 3
    return smoothed


def clip_values(array):
    clipped = np.clip(array, 0, 1)
    return clipped


original = np.random.rand(100)
processed = preprocess_data(original)

img = np.random.rand(64, 64)
binary = apply_threshold(img)
