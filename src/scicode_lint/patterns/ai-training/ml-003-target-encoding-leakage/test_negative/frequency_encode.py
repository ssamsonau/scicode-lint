def encode_categories(df):
    for col in df.select_dtypes(include=["object"]).columns:
        counts = df[col].value_counts()
        df[f"{col}_encoded"] = df[col].map(counts)
    return df


def label_encode_train_test(train_df, test_df):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(train_df["category"])
    train_df["cat_enc"] = le.transform(train_df["category"])
    test_df["cat_enc"] = le.transform(test_df["category"])
    return train_df, test_df
