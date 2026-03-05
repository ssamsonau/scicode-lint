import threading

import numpy as np


def normalize_features(data, result_dict, idx):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized = (data - mean) / (std + 1e-8)
    result_dict[idx] = normalized


def batch_normalize(dataset_chunks):
    threads = []
    results = {}

    for i, chunk in enumerate(dataset_chunks):
        t = threading.Thread(target=normalize_features, args=(chunk, results, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return [results[i] for i in range(len(dataset_chunks))]


data = np.random.randn(10000, 100)
chunks = np.array_split(data, 4)
normalized_data = batch_normalize(chunks)
