import numpy as np


def preprocess_data(data):
    data -= data.mean()
    data /= data.std() + 1e-8
    return data


def apply_threshold(image):
    image[image < 0.5] = 0
    image[image >= 0.5] = 1
    return image


def smooth_signal(signal):
    for i in range(1, len(signal) - 1):
        signal[i] = (signal[i - 1] + signal[i] + signal[i + 1]) / 3
    return signal


def clip_values(array):
    array[array < 0] = 0
    array[array > 1] = 1
    return array


original = np.random.rand(100)
processed = preprocess_data(original)

img = np.random.rand(64, 64)
binary = apply_threshold(img)
