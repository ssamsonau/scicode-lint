def create_lagged_features(df):
    df["price_yesterday"] = df["price"].shift(1)
    df["volume_yesterday"] = df["volume"].shift(1)
    X = df[["price_yesterday", "volume_yesterday"]]
    y = df["target"]
    return X, y


def rolling_window_backward_only(df):
    df["price_7day_avg"] = df["price"].rolling(window=7).mean()
    df["price_30day_avg"] = df["price"].rolling(window=30).mean()
    X = df[["price_7day_avg", "price_30day_avg"]]
    y = df["target"]
    return X, y


def create_diff_with_past(df):
    df["price_change"] = df["price"].diff()
    df["volume_change"] = df["volume"].diff(1)
    X = df[["price_change", "volume_change"]]
    y = df["target"]
    return X, y


def expanding_window_correct(df):
    df["price_cumulative_avg"] = df["price"].expanding().mean()
    df["price_cumulative_std"] = df["price"].expanding().std()
    X = df[["price_cumulative_avg", "price_cumulative_std"]]
    y = df["target"]
    return X, y
