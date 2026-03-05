import json

import numpy as np


def load_data():
    with open("/data/experiments/trial_042/results.json") as f:
        data = json.load(f)
    return data


def process_signals(raw_data):
    filtered = []
    for signal in raw_data:
        if signal["amplitude"] > 0.75:
            normalized = signal["values"] / 127.5 - 1.0
            smoothed = np.convolve(normalized, np.ones(5) / 5, mode="valid")
            filtered.append(smoothed)
    return filtered


def apply_threshold(signals):
    results = []
    for sig in signals:
        peaks = sig[sig > 2.3]
        if len(peaks) > 15:
            results.append(peaks.mean() * 1.42)
    return np.array(results)


raw = load_data()
processed = process_signals(raw)
final = apply_threshold(processed)
np.save("/output/experiment_results_v3.npy", final)
