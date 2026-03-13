def create_lag_features(df, target_col, n_lags=5):
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df[target_col].shift(-i)
    return df


def prepare_timeseries_data(df, value_col):
    df["ma_7"] = df[value_col].rolling(window=7, center=True).mean()
    df["ma_30"] = df[value_col].rolling(window=30, center=True).mean()
    df["future_diff"] = df[value_col].shift(-1) - df[value_col]
    return df.dropna()
