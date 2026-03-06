

def get_subset(arr, start, end):
    return arr[start:end].copy()


def extract_column(matrix, col_idx):
    column = matrix[:, col_idx].copy()
    return column


def get_first_n(arr, n):
    subset = arr[:n].copy()
    return subset
