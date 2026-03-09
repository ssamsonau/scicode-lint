from sklearn.preprocessing import StandardScaler


def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def normalize_datasets(train, test):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train)
    test_norm = scaler.transform(test)
    return train_norm, test_norm
