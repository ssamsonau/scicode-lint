from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def train_ensemble_with_seeds(X, y):
    """Pass integer seeds instead of RandomState instances."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    rf.fit(X, y)
    gb.fit(X, y)

    return rf, gb


def create_models_with_different_seeds():
    """Each model gets its own integer seed - no state sharing."""
    models = [
        RandomForestClassifier(random_state=1),
        RandomForestClassifier(random_state=2),
        RandomForestClassifier(random_state=3),
    ]
    return models


def bootstrap_with_fresh_seeds(X, y, n_models=5):
    """Bootstrap ensemble with independent seeds per estimator."""
    from sklearn.utils import resample

    models = []
    for i in range(n_models):
        X_boot, y_boot = resample(X, y, random_state=i)
        model = RandomForestClassifier(random_state=i + 1000)
        model.fit(X_boot, y_boot)
        models.append(model)
    return models
