import numpy as np
import pandas as pd


def create_features_with_future_shift(df):
    future_prices = pd.Series(df["price"].values[1:].tolist() + [np.nan])
    future_volumes = pd.Series(df["volume"].values[1:].tolist() + [np.nan])
    df["price_next_day"] = future_prices.values
    df["volume_next_day"] = future_volumes.values
    X = df[["price_next_day", "volume_next_day"]]
    y = df["target"]
    return X, y


def rolling_window_with_future(df):
    df["price_centered_avg"] = df["price"].rolling(window=5, center=True).mean()
    X = df[["price_centered_avg"]]
    y = df["target"]
    return X, y


def create_diff_with_future(df):
    price_array = df["price"].values
    forward_changes = np.append(price_array[1:] - price_array[:-1], np.nan)
    df["price_change_future"] = forward_changes
    X = df[["price_change_future"]]
    y = df["target"]
    return X, y
