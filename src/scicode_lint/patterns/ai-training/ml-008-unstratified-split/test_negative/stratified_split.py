from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def create_holdout(features, labels, ratio=0.3):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
    train_idx, test_idx = next(sss.split(features, labels))
    return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]
