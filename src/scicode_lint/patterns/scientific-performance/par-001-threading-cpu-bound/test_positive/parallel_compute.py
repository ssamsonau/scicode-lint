import threading

import numpy as np


def parallel_matrix_multiply(A, B, num_threads=4):
    result = np.zeros((A.shape[0], B.shape[1]))
    threads = []
    chunk_size = A.shape[0] // num_threads

    def worker(start, end):
        result[start:end] = A[start:end] @ B

    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else A.shape[0]
        t = threading.Thread(target=worker, args=(start, end))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    return result
