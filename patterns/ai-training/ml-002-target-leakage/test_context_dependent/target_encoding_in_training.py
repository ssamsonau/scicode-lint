import pandas as pd
from sklearn.model_selection import train_test_split


def target_encoding_on_train_only(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[["feature1", "category"]], df["target"], test_size=0.2, random_state=42
    )

    train_df = pd.DataFrame({"category": X_train["category"], "target": y_train})
    category_means = train_df.groupby("category")["target"].mean()

    X_train["category_target_mean"] = X_train["category"].map(category_means)
    X_test["category_target_mean"] = X_test["category"].map(category_means)

    return X_train, X_test, y_train, y_test
