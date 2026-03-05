from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def preprocess_test_with_fit_transform(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    normalizer = StandardScaler()
    X_train_scaled = normalizer.fit_transform(X_train)
    X_test_scaled = normalizer.fit_transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def separate_scalers_for_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_scaler = MinMaxScaler()
    X_train_scaled = train_scaler.fit_transform(X_train)
    test_scaler = MinMaxScaler()
    X_test_scaled = test_scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def refit_scaler_on_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = RobustScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)
    preprocessor.fit(X_test)
    X_test_scaled = preprocessor.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
