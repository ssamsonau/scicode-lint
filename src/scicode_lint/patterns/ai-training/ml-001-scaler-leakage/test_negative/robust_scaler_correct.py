from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_with_pipeline(X, y):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    return scores.mean()


def fit_pipeline(X_train, y_train, X_holdout):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_holdout)
    return predictions
