def weighted_average(values, weights):
    return (values * weights.reshape(-1)).sum() / weights.sum()


def apply_weights(matrix, row_weights, col_weights):
    weighted = matrix * row_weights.reshape(-1, 1) * col_weights.reshape(1, -1)
    return weighted


def scale_features(X, scales):
    return X * scales.reshape(1, -1)
