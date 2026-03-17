from concurrent.futures import ProcessPoolExecutor

import numpy as np


def clip_value(x):
    if x < -1.0:
        return -1.0
    elif x > 1.0:
        return 1.0
    return x


def parallel_clip(array, max_workers=4):
    flat = array.flatten().tolist()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(clip_value, flat))
    return np.array(results).reshape(array.shape)


matrix = np.random.randn(200, 300)
clipped = parallel_clip(matrix)
