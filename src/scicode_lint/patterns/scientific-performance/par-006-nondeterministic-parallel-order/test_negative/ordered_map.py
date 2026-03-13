from concurrent.futures import ProcessPoolExecutor

import numpy as np


def compute_feature(x):
    return np.log(abs(x) + 1) * np.sign(x)


def parallel_ordered_computation(values):
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compute_feature, values))
    return np.array(results)


data = np.random.randn(8000)
features = parallel_ordered_computation(data)
