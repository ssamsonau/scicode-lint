from sklearn.preprocessing import StandardScaler


def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    X_train_scaled = scaler.transform(X_train)
    return X_train_scaled, X_test_scaled


def normalize_datasets(train, test):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    test_norm = scaler.fit_transform(test)
    train_norm = scaler.transform(train)
    return train_norm, test_norm
