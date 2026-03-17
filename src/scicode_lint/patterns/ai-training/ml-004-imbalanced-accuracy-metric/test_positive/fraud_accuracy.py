from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FraudDetectionPipeline:
    """Fraud detection pipeline with accuracy-based evaluation."""

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", IsolationForest(contamination=0.01, random_state=42)),
            ]
        )
        self.accuracy = None

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.pipeline.fit(X_train)
        predictions = self.pipeline.predict(X_test)
        predictions = (predictions == -1).astype(int)
        self.accuracy = accuracy_score(y_test, predictions)
        return self.accuracy


def tune_anomaly_detector(X, y):
    """Hyperparameter tuning for anomaly detection using accuracy scoring."""
    param_grid = {"contamination": [0.01, 0.02, 0.05], "n_estimators": [50, 100, 200]}

    def anomaly_accuracy(estimator, X, y_true):
        y_pred = (estimator.predict(X) == -1).astype(int)
        return accuracy_score(y_true, y_pred)

    grid = GridSearchCV(
        IsolationForest(random_state=0), param_grid, scoring=make_scorer(anomaly_accuracy), cv=3
    )
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_
