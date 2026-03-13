import pandas as pd
from sklearn.model_selection import train_test_split


def create_target_encoding_correctly(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1)
    target_means = train_df.groupby("category")[target_col].mean()
    X_train["category_target_mean"] = X_train["category"].map(target_means)
    X_test["category_target_mean"] = X_test["category"].map(target_means)
    X_test["category_target_mean"].fillna(y_train.mean(), inplace=True)
    return X_train, X_test, y_train, y_test


def no_target_derived_features(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
