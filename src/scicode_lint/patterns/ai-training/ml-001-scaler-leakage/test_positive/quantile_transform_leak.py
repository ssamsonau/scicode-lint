from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR


def build_regression_model(X, y, test_size=0.2):
    scaler = QuantileTransformer(output_distribution="normal")
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    model = SVR()
    model.fit(X_train, y_train)
    return model, X_test, y_test
