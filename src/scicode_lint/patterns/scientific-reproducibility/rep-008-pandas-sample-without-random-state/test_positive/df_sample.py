def sample_data(df, frac=0.1):
    return df.sample(frac=frac)


def bootstrap_sample(df, n=1000):
    return df.sample(n=n, replace=True)


def stratified_sample(df, column, frac=0.2):
    return df.groupby(column).apply(lambda x: x.sample(frac=frac))
