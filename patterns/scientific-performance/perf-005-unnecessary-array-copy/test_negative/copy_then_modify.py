

def normalize_data(arr):
    data = arr.copy()
    data -= data.mean()
    data /= data.std()
    return data


def zero_negative_values(arr):
    result = arr.copy()
    result[result < 0] = 0
    return result


def scale_array(arr, factor):
    scaled = arr.copy()
    scaled *= factor
    return scaled
