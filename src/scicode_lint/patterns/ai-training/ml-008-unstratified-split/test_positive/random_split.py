from sklearn.model_selection import train_test_split


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_holdout(features, labels, ratio=0.3):
    n_test = int(len(features) * ratio)
    indices = list(range(len(features)))
    import random

    random.shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]
