import multiprocessing as mp

import numpy as np


def parallel_matrix_multiply(A, B, num_processes=4):
    def worker(args):
        start, end, A_chunk, B = args
        return start, A_chunk @ B

    chunk_size = A.shape[0] // num_processes
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_processes - 1 else A.shape[0]
        chunks.append((start, end, A[start:end], B))

    with mp.Pool(num_processes) as pool:
        results = pool.map(worker, chunks)

    result = np.zeros((A.shape[0], B.shape[1]))
    for start, chunk_result in results:
        end = start + len(chunk_result)
        result[start:end] = chunk_result
    return result
