import numpy as np


def normalize_inplace(data):
    data -= data.mean(axis=0)
    data /= data.std(axis=0).clip(min=1e-8)
    return data


def scale_and_shift(arr, scale, offset):
    result = arr.copy()
    result *= scale
    result += offset
    np.clip(result, 0, 1, out=result)
    return result


def exponential_moving_average(signal, alpha=0.3):
    ema = np.empty_like(signal)
    ema[0] = signal[0]
    for i in range(1, len(signal)):
        ema[i] = alpha * signal[i] + (1 - alpha) * ema[i - 1]
    return ema
