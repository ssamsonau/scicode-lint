import numpy as np


def create_normalized_version(data):
    """Create new array via numpy operations - naturally non-mutating."""
    mean_val = np.mean(data)
    std_val = np.std(data) + 1e-8
    return (data - mean_val) / std_val


def binarize_image(image, threshold=0.5):
    """Use np.where for thresholding - creates new array."""
    return np.where(image >= threshold, 1.0, 0.0)


def convolve_signal(signal, kernel):
    """Signal smoothing via convolution - doesn't modify input."""
    return np.convolve(signal, kernel, mode="same")


def bounded_values(array, low=0, high=1):
    """Use np.clip which returns new array."""
    return np.clip(array, low, high)


def cumulative_sum(data):
    """Cumulative operations return new arrays."""
    return np.cumsum(data)


def element_wise_max(arr1, arr2):
    """np.maximum returns new array."""
    return np.maximum(arr1, arr2)


original = np.random.rand(100)
normalized = create_normalized_version(original)
assert np.allclose(original.mean(), 0.5, atol=0.2)
