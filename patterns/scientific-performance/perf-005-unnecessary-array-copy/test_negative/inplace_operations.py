def normalize_data(arr):
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / std


def process_array(input_arr):
    return input_arr * 2 + 1
