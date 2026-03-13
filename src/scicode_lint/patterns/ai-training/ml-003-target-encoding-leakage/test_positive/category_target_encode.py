def encode_categories(df, target_col):
    """Encodes categorical columns using target mean - applied to full dataset."""
    for col in df.select_dtypes(include=["object"]).columns:
        means = df.groupby(col)[target_col].mean()
        df[f"{col}_encoded"] = df[col].map(means)
    return df


def preprocess_data(raw_df, target_col):
    """Preprocessing pipeline that applies target encoding before any split."""
    df = raw_df.copy()
    df = encode_categories(df, target_col)
    return df
