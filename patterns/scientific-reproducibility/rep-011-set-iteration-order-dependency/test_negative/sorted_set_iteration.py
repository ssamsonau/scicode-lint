import numpy as np


def process_unique_items(items):
    unique = set(items)
    results = []
    for item in sorted(unique):
        results.append(item * 2)
    return results


def select_features(X, feature_set):
    selected = []
    for feature in sorted(feature_set):
        selected.append(X[:, feature])
    return np.column_stack(selected)


def build_vocabulary(texts):
    vocab = set()
    for text in texts:
        vocab.update(text.split())

    word_to_idx = {}
    for i, word in enumerate(sorted(vocab)):
        word_to_idx[word] = i
    return word_to_idx


def write_feature_names(features, filepath):
    feature_set = set(features)
    with open(filepath, "w") as f:
        for name in sorted(feature_set):
            f.write(name + "\n")
