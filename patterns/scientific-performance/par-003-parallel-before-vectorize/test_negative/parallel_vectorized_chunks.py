from multiprocessing import Pool

import numpy as np


def vectorized_chunk_process(chunk):
    result = np.sqrt(np.abs(chunk)) * np.sign(chunk)
    return np.sum(result)


def parallel_on_vectorized_chunks(data, num_workers):
    chunks = np.array_split(data, num_workers)
    with Pool(processes=num_workers) as pool:
        results = pool.map(vectorized_chunk_process, chunks)
    return sum(results)


large_array = np.random.randn(1000000)
total = parallel_on_vectorized_chunks(large_array, 4)
