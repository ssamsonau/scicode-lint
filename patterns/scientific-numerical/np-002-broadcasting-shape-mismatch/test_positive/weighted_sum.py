def weighted_average(values, weights):
    return (values * weights).sum() / weights.sum()


def apply_weights(matrix, row_weights, col_weights):
    weighted = matrix * row_weights * col_weights
    return weighted


def scale_features(X, scales):
    return X * scales
