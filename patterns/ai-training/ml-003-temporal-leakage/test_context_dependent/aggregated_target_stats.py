from sklearn.model_selection import train_test_split


def use_historical_averages(df, target_col="target"):
    category_default_rates = df.groupby("category")[target_col].mean()
    df["category_avg_target"] = df["category"].map(category_default_rates)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
