import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


class StratifiedEvaluator:
    def __init__(self, n_folds=5):
        self.cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

    def cross_validate(self, X, y):
        fold_scores = []
        for train_idx, val_idx in self.cv.split(X, y):
            clf = SVC(kernel="rbf", class_weight="balanced")
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[val_idx])
            fold_scores.append(f1_score(y[val_idx], preds, average="macro"))
        return np.mean(fold_scores), np.std(fold_scores)


def nested_stratified_cv(X, y, param_grid):
    from sklearn.model_selection import GridSearchCV

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = []
    for train_idx, test_idx in outer_cv.split(X, y):
        search = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring="f1_macro")
        search.fit(X[train_idx], y[train_idx])
        score = search.score(X[test_idx], y[test_idx])
        results.append(score)
    return results
