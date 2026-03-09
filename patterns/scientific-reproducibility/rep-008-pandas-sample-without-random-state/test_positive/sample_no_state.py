def subsample_data(df, frac=0.1):
    return df.sample(frac=frac)


def shuffle_dataframe(df):
    return df.sample(frac=1.0)


def bootstrap_sample(df, n_samples):
    return df.sample(n=n_samples, replace=True)


def get_validation_set(df, val_size=0.2):
    val_df = df.sample(frac=val_size)
    train_df = df.drop(val_df.index)
    return train_df, val_df
