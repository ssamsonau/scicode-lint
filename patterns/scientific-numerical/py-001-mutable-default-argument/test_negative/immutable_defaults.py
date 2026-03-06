import numpy as np


def compute_result(x, scale=1.0):
    return x * scale


def process_data(data, threshold=0.5):
    filtered = data[data > threshold]
    return filtered


def optimize(loss, learning_rate=0.01, momentum=0.9):
    update = learning_rate * loss + momentum
    return update


def analyze_signal(signal, window_size=10):
    windowed = signal[:window_size]
    return np.mean(windowed)


result1 = compute_result(5.0)
result2 = compute_result(5.0, scale=2.0)

data = np.random.rand(100)
filtered = process_data(data)
