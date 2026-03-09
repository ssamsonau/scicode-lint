def create_lag_features(df):
    df["prev_price"] = df["price"].shift(1)
    df["past_avg"] = df["price"].shift(1).rolling(3).mean()
    return df.dropna()


def add_lag_indicators(df, target_col):
    df["yesterday"] = df[target_col].shift(1)
    df["last_week_avg"] = df[target_col].shift(1).rolling(7).mean()
    return df
