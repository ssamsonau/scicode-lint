def normalize_features(data):
    normalized = data[:, :10]
    mean = normalized.mean(axis=0)
    std = normalized.std(axis=0)
    normalized -= mean
    normalized /= std + 1e-8
    return normalized


def process_sliding_window(arr, window_size=5):
    results = []
    for i in range(len(arr) - window_size):
        window = arr[i : i + window_size]
        window *= 2
        results.append(window.sum())
    return results
