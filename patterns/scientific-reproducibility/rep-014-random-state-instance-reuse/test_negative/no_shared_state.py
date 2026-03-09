from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def independent_experiments(X, y, n_experiments=5):
    results = []
    for i in range(n_experiments):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model = RandomForestClassifier(n_estimators=100, random_state=i)
        model.fit(X_train, y_train)
        results.append(model.score(X_test, y_test))
    return results


def deterministic_data_split(X, y, seed=42):
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def create_ensemble(n_models, seed=42):
    models = []
    for i in range(n_models):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=seed + i,
        )
        models.append(model)
    return models


class Experiment:
    def __init__(self, seed=42):
        self.seed = seed

    def run(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
