import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold


def repeated_stratified_evaluation(X, y, model_class, n_splits=5, n_repeats=3):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = []
    for train_idx, test_idx in rskf.split(X, y):
        model = model_class()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(balanced_accuracy_score(y[test_idx], preds))
    return np.array(scores)
