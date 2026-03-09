def process_unique_items(items):
    unique = set(items)
    results = []
    for item in unique:
        results.append(item * 2)
    return results


def get_unique_features(feature_dict):
    features = set(feature_dict.keys())
    return list(features)


def dedupe_and_process(data):
    seen = set()
    output = []
    for item in data:
        if item not in seen:
            seen.add(item)
    for s in seen:
        output.append(s)
    return output
