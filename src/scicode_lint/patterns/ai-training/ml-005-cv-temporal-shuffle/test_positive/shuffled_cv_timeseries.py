from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


def cross_validate_with_shuffle(df):
    X = df[["feature1", "feature2", "feature3"]]
    y = df["target"]
    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"Cross-validation scores: {scores}")
    print(f"Mean R²: {scores.mean():.3f}")
    return scores


def evaluate_stock_prediction_shuffled(df):
    df = df.sort_values("date")
    X = df[["open", "high", "low", "volume"]]
    y = df["close"]
    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    print(f"MSE scores: {-scores}")
    return scores
