def vectorized_square(arr):
    return arr**2


def vectorized_normalize(data):
    return (data - data.mean()) / data.std()
