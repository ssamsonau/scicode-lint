import numpy as np


def process_data(data):
    temp1 = data * 2
    temp2 = temp1 + 10
    temp3 = np.sqrt(temp2)
    temp4 = temp3 / np.max(temp3)
    return temp4


def compute_rms_energy(signal, frame_size=1024):
    frames = signal[: len(signal) // frame_size * frame_size].reshape(-1, frame_size)
    squared = frames**2
    mean_power = np.mean(squared, axis=1)
    rms = np.sqrt(mean_power)
    db_values = 20 * np.log10(rms + 1e-10)
    return db_values
