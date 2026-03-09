from sklearn.model_selection import train_test_split


def prepare_data_correctly(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def reproducible_split(data, labels):
    SEED = 123
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.3, random_state=SEED
    )
    return train_x, test_x, train_y, test_y
