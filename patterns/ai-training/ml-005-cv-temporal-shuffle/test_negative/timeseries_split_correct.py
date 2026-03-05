from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score


def cross_validate_timeseries_split(df):
    df = df.sort_values("date")
    X = df[["feature1", "feature2", "feature3"]]
    y = df["target"]
    model = LinearRegression()
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"Time-series CV scores: {scores}")
    print(f"Mean R²: {scores.mean():.3f}")
    return scores


def manual_time_based_split(df):
    df = df.sort_values("date")
    split_date = df["date"].quantile(0.8)
    train = df[df["date"] <= split_date]
    test = df[df["date"] > split_date]
    X_train = train[["feature1", "feature2", "feature3"]]
    y_train = train["target"]
    X_test = test[["feature1", "feature2", "feature3"]]
    y_test = test["target"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test R²: {score:.3f}")
    return score


def kfold_without_shuffle_on_timeseries(df):
    df = df.sort_values("date")
    X = df[["feature1", "feature2", "feature3"]]
    y = df["target"]
    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=False)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return scores
