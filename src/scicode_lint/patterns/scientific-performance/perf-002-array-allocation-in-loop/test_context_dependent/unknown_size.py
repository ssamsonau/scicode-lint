import numpy as np


def process_stream(data_stream, stop_condition):
    results = []
    for data in data_stream:
        if stop_condition(data):
            break
        processed = np.array(data)
        results.append(processed)
    return np.vstack(results) if results else np.array([])


def filter_and_collect(items, predicate):
    collected = []
    for item in items:
        if predicate(item):
            arr = np.array(item.values)
            collected.append(arr)
    return collected
