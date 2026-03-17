"""Using sorted(set) for reproducible iteration order."""

import numpy as np


def select_features_sorted(X, feature_set):
    """Select features from set in sorted order for reproducibility."""
    selected = []
    for feature in sorted(feature_set):
        selected.append(X[:, feature])
    return np.column_stack(selected)


def write_unique_labels(labels, filepath):
    """Write unique labels in sorted order."""
    unique = set(labels)
    with open(filepath, "w") as f:
        for label in sorted(unique):
            f.write(f"{label}\n")


def get_vocabulary(texts: list[str]) -> dict[str, int]:
    """Build vocabulary with deterministic word-to-index mapping."""
    all_words = set()
    for text in texts:
        all_words.update(text.split())
    return {word: idx for idx, word in enumerate(sorted(all_words))}
