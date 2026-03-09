def create_lag_features(df, target_col, n_lags=5):
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df[target_col].shift(i)
    return df


def prepare_timeseries_data(df, value_col):
    df["ma_7"] = df[value_col].rolling(window=7).mean()
    df["ma_30"] = df[value_col].rolling(window=30).mean()
    df["past_diff"] = df[value_col] - df[value_col].shift(1)
    return df.dropna()
