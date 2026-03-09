from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold


def get_cv_splitter(n_splits=5):
    return KFold(n_splits=n_splits, shuffle=True)


def stratified_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True)


def shuffle_split_cv(n_splits=10, test_size=0.2):
    return ShuffleSplit(n_splits=n_splits, test_size=test_size)
