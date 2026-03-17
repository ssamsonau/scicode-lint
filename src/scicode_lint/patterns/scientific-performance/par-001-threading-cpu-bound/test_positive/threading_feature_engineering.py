import math
import random
from concurrent.futures import ThreadPoolExecutor


def compute_statistics(data_batch):
    features = []
    for series in data_batch:
        n = len(series)
        mean = sum(series) / n
        variance = sum((x - mean) ** 2 for x in series) / n
        std = math.sqrt(variance)
        skewness = sum((x - mean) ** 3 for x in series) / (n * std**3) if std > 0 else 0.0
        sorted_vals = sorted(series)
        median = sorted_vals[n // 2]
        features.append([mean, std, skewness, median])
    return features


def process_dataset(batches, max_threads=8):
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(compute_statistics, batch) for batch in batches]
        results = [f.result() for f in futures]
    return [item for sublist in results for item in sublist]


data_batches = [[[random.gauss(0, 1) for _ in range(500)] for _ in range(50)] for _ in range(8)]
all_features = process_dataset(data_batches)
