import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("/data/experiments")
OUTPUT_DIR = Path("/output")
EXPERIMENT_ID = "trial_042"

AMPLITUDE_THRESHOLD = 0.75
NORMALIZATION_FACTOR = 127.5
SMOOTHING_WINDOW = 5
PEAK_THRESHOLD = 2.3
MIN_PEAK_COUNT = 15
SCALING_FACTOR = 1.42


def load_data():
    data_path = DATA_DIR / EXPERIMENT_ID / "results.json"
    with open(data_path) as f:
        data = json.load(f)
    return data


def process_signals(raw_data):
    filtered = []
    for signal in raw_data:
        if signal["amplitude"] > AMPLITUDE_THRESHOLD:
            normalized = signal["values"] / NORMALIZATION_FACTOR - 1.0
            kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
            smoothed = np.convolve(normalized, kernel, mode="valid")
            filtered.append(smoothed)
    return filtered


def apply_threshold(signals):
    results = []
    for sig in signals:
        peaks = sig[sig > PEAK_THRESHOLD]
        if len(peaks) > MIN_PEAK_COUNT:
            results.append(peaks.mean() * SCALING_FACTOR)
    return np.array(results)


raw = load_data()
processed = process_signals(raw)
final = apply_threshold(processed)

output_path = OUTPUT_DIR / f"experiment_results_{EXPERIMENT_ID}.npy"
np.save(output_path, final)
