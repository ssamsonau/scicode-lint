def sample_with_seed(df, n=100, seed=42):
    """Sampling with random_state for reproducibility."""
    return df.sample(n=n, random_state=seed)


def bootstrap_sample(df, frac=1.0, random_state=42):
    """Bootstrap sampling with fixed random state."""
    return df.sample(frac=frac, replace=True, random_state=random_state)


def stratified_subsample(df, group_col, frac=0.1, seed=42):
    """Stratified sampling with reproducible random state."""
    return df.groupby(group_col).apply(lambda x: x.sample(frac=frac, random_state=seed))


def split_data_reproducibly(df, train_frac=0.8, random_state=42):
    """Train/test split using pandas sample with random_state."""
    train = df.sample(frac=train_frac, random_state=random_state)
    test = df.drop(train.index)
    return train, test
