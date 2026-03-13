def create_lag_features(df):
    df["next_price"] = df["price"].shift(-1)
    df["future_avg"] = df["price"].shift(-3).rolling(3).mean()
    return df.dropna()


def add_future_indicators(df, target_col):
    df["tomorrow"] = df[target_col].shift(-1)
    df["next_week_avg"] = df[target_col].shift(-7).rolling(7).mean()
    return df
