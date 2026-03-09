def encode_categories(df, target_col):
    for col in df.select_dtypes(include=["object"]).columns:
        means = df.groupby(col)[target_col].mean()
        df[f"{col}_encoded"] = df[col].map(means)
    return df


def target_encode_train_test(train_df, test_df, target):
    mapping = train_df.groupby("category")[target].mean().to_dict()
    train_df["cat_enc"] = train_df["category"].map(mapping)
    test_df["cat_enc"] = test_df["category"].map(mapping)
    return train_df, test_df
