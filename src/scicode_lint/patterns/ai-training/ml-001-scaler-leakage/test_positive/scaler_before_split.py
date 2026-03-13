import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_with_test_stats(train_data, test_data):
    full_dataset = torch.vstack([train_data, test_data])
    dataset_mean = full_dataset.mean()
    dataset_std = full_dataset.std()
    train_normalized = (train_data - dataset_mean) / dataset_std
    test_normalized = (test_data - dataset_mean) / dataset_std
    return train_normalized, test_normalized


def fit_scaler_before_split(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def normalize_concat_data(train_data, test_data):
    merged_arrays = np.vstack([train_data, test_data])
    normalizer = MinMaxScaler()
    normalized_all = normalizer.fit_transform(merged_arrays)
    split_point = len(train_data)
    train_scaled = normalized_all[:split_point]
    test_scaled = normalized_all[split_point:]
    return train_scaled, test_scaled
