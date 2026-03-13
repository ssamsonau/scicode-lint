import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


def process_chunks(data, num_workers=4):
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_arr[:] = data[:]

    def worker(args):
        shm_name, shape, dtype, start, end = args
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        arr = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        result = arr[start:end].sum()
        existing_shm.close()
        return result

    chunk_size = len(data) // num_workers
    args = [
        (shm.name, data.shape, data.dtype, i * chunk_size, (i + 1) * chunk_size)
        for i in range(num_workers)
    ]

    with mp.Pool(num_workers) as pool:
        results = pool.map(worker, args)

    shm.close()
    shm.unlink()
    return sum(results)
