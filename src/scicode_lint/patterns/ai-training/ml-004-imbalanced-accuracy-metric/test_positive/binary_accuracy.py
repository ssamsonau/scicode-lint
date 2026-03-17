from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


class FraudClassifier:
    """Fraud detection classifier using cross-validation."""

    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.accuracy = None

    def cross_validate(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")
        self.accuracy = scores.mean()
        return self.accuracy


class AnomalyClassifier:
    """Anomaly detection using gradient boosting with accuracy metric."""

    def __init__(self):
        self.model = GradientBoostingClassifier()
        self.accuracy = None

    def evaluate(self, X, y):
        self.accuracy = cross_val_score(self.model, X, y, cv=3, scoring="accuracy").mean()
        return self.accuracy
