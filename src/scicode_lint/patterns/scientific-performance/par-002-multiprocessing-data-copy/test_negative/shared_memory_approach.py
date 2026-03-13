from multiprocessing import Pool, shared_memory

import numpy as np


def process_chunk(args):
    shm_name, shape, dtype, start_idx, end_idx = args
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    result = np.sum(arr[start_idx:end_idx])
    existing_shm.close()
    return result


def parallel_compute_with_shared_memory(data):
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]

    chunk_size = len(data) // 4
    args_list = [
        (shm.name, data.shape, data.dtype, i * chunk_size, (i + 1) * chunk_size) for i in range(4)
    ]

    with Pool(processes=4) as pool:
        results = pool.map(process_chunk, args_list)

    shm.close()
    shm.unlink()
    return sum(results)


large_data = np.random.randn(100000, 1000)
total = parallel_compute_with_shared_memory(large_data)
