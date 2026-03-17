import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures


def create_interaction_features(X_train, X_test, degree=2):
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    return X_train_poly, X_test_poly


def reduce_dimensions(X_train, X_test, n_components=10):
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    explained = np.sum(pca.explained_variance_ratio_)
    return X_train_reduced, X_test_reduced, explained
