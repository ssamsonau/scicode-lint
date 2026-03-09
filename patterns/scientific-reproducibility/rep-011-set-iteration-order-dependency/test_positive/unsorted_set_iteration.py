import numpy as np


def select_features(X, feature_set):
    selected = []
    for feature in feature_set:
        selected.append(X[:, feature])
    return np.column_stack(selected)


def get_unique_labels(labels):
    unique = set(labels)
    return list(unique)


def write_feature_names(features, filepath):
    feature_set = set(features)
    with open(filepath, "w") as f:
        for name in feature_set:
            f.write(f"{name}\n")


if __name__ == "__main__":
    X = np.random.randn(100, 10)
    features = {0, 2, 5, 7}
    result = select_features(X, features)
    print(f"Selected shape: {result.shape}")
