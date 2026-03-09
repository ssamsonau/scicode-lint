import pandas as pd


def subsample_data(df):
    subset = df.sample(n=1000)
    return subset


def create_balanced_sample(df, n_per_class=100):
    samples = []
    for label in df["label"].unique():
        class_df = df[df["label"] == label]
        sample = class_df.sample(n=n_per_class)
        samples.append(sample)
    return pd.concat(samples)


def random_train_test(df, train_frac=0.8):
    train = df.sample(frac=train_frac)
    test = df.drop(train.index)
    return train, test


if __name__ == "__main__":
    df = pd.DataFrame({"x": range(10000), "label": [0, 1] * 5000})
    subset = subsample_data(df)
    print(f"Sampled {len(subset)} rows")
