from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def evaluate_with_embedded_selection(X, y, n_folds=5):
    pipe = Pipeline(
        [
            ("selector", SelectFromModel(GradientBoostingClassifier(n_estimators=50))),
            ("classifier", GradientBoostingClassifier(n_estimators=200)),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=n_folds, scoring="f1_weighted")
    return scores


class ColumnDropper:
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")
