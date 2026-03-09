from sklearn.model_selection import train_test_split


def encode_categories(df, target_col, cat_cols):
    for col in cat_cols:
        means = df.groupby(col)[target_col].mean()
        df[f"{col}_encoded"] = df[col].map(means)
    return df


def prepare_data(df, target_col, categorical_cols):
    for col in categorical_cols:
        encoding = df.groupby(col)[target_col].mean().to_dict()
        df[f"{col}_target_enc"] = df[col].map(encoding)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2)
