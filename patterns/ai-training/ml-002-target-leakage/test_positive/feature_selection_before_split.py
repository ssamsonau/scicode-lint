import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


def select_features_before_split(X, y):
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def correlation_filter_before_split(X, y):
    correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    top_features = np.argsort(correlations)[-10:]
    X_filtered = X[:, top_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
