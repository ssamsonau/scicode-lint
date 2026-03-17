import multiprocessing as mp
import tempfile

import numpy as np


def worker_memmap(args):
    filepath, shape, dtype, start, end = args
    arr = np.memmap(filepath, dtype=dtype, mode="r", shape=shape)
    result = arr[start:end].sum()
    del arr
    return result


def parallel_compute_with_memmap(data, num_workers=4):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filepath = f.name
        fp = np.memmap(filepath, dtype=data.dtype, mode="w+", shape=data.shape)
        fp[:] = data[:]
        fp.flush()
        del fp

    chunk_size = len(data) // num_workers
    args = [
        (filepath, data.shape, data.dtype, i * chunk_size, (i + 1) * chunk_size)
        for i in range(num_workers)
    ]

    with mp.Pool(num_workers) as pool:
        results = pool.map(worker_memmap, args)

    return sum(results)
