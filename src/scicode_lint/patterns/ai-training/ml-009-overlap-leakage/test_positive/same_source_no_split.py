from sklearn.linear_model import RidgeClassifier


def quality_based_split(df, quality_col="quality_score"):
    high_quality = df[df[quality_col] > 0.5]
    low_quality = df[df[quality_col] > 0.3]

    X_train = high_quality.drop(columns=["target", quality_col]).values
    y_train = high_quality["target"].values
    X_test = low_quality.drop(columns=["target", quality_col]).values
    y_test = low_quality["target"].values

    model = RidgeClassifier()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def confidence_split(predictions_df):
    confident = predictions_df[predictions_df["confidence"] >= 0.7]
    uncertain = predictions_df[predictions_df["confidence"] >= 0.4]

    train_features = confident[["f1", "f2", "f3"]].values
    train_labels = confident["label"].values
    test_features = uncertain[["f1", "f2", "f3"]].values
    test_labels = uncertain["label"].values

    return train_features, test_features, train_labels, test_labels
