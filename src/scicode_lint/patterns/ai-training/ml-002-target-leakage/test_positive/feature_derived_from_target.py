from sklearn.model_selection import train_test_split


def create_target_derived_feature(df):
    df["category_mean_target"] = df.groupby("category")["target"].transform("mean")

    X = df[["feature1", "feature2", "category_mean_target"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def encode_target_as_feature(df):
    df["target_category"] = (df["target"] > df["target"].median()).astype(int)

    X = df[["feature1", "target_category"]]
    y = df["target"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
