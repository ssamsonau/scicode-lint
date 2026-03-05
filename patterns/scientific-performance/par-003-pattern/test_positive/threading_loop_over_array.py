from concurrent.futures import ThreadPoolExecutor

import numpy as np


def square_if_positive(value):
    if value > 0:
        return value**2
    return 0


def parallel_conditional_square(array):
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(square_if_positive, x) for x in array]
        for future in futures:
            results.append(future.result())
    return np.array(results)


input_array = np.random.randn(50000)
output = parallel_conditional_square(input_array)
