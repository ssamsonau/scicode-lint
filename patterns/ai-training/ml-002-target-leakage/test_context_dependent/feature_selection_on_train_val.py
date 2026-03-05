import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def select_features_on_train_val(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    selector = SelectKBest(f_classif, k=10)
    X_train_val_selected = selector.fit_transform(X_train_val, y_train_val)
    X_test_selected = selector.transform(X_test)
    n_train = len(X_train)
    X_train_selected = X_train_val_selected[:n_train]
    X_val_selected = X_train_val_selected[n_train:]
    return X_train_selected, X_val_selected, X_test_selected
