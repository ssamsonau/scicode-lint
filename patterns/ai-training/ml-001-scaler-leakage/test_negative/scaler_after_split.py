def normalize_correctly(train_data, test_data):
    mean = train_data.mean()
    std = train_data.std()
    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std
    return train_normalized, test_normalized


def normalize_with_sklearn_pipeline(train_data, test_data):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_normalized = scaler.transform(train_data)
    test_normalized = scaler.transform(test_data)
    return train_normalized, test_normalized
