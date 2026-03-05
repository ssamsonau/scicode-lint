import numpy as np
from sklearn.model_selection import train_test_split


def create_target_mean_encoding(df, target_col="target"):
    category_stats = df.groupby("category").agg({target_col: "mean"})
    df["encoded_category"] = df["category"].apply(lambda x: category_stats.loc[x, target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def add_global_target_stats(X, y):
    overall_mean = np.mean(y)
    overall_variance = np.var(y)
    X["avg_target"] = overall_mean
    X["var_target"] = overall_variance
    X["standardized_y"] = (y - overall_mean) / np.sqrt(overall_variance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
