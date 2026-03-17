from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


class ModelSelector:
    def __init__(self, param_grid, cv=5):
        self.param_grid = param_grid
        self.cv = cv
        self.best_model_ = None

    def fit_and_evaluate(self, X, y, test_size=0.2):
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        search = GridSearchCV(
            GradientBoostingClassifier(),
            self.param_grid,
            cv=self.cv,
            scoring="f1_macro",
        )
        search.fit(X_dev, y_dev)
        self.best_model_ = search.best_estimator_

        holdout_preds = self.best_model_.predict(X_holdout)
        report = classification_report(y_holdout, holdout_preds, output_dict=True)
        return report
