import queue
import threading


def multiply_matrices(A, B, result_queue, idx):
    rows_a = len(A)
    cols_b = len(B[0])
    cols_a = len(A[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0.0
            for k in range(cols_a):
                total += A[i][k] * B[k][j]
            result[i][j] = total
    result_queue.put((idx, result))


def parallel_matrix_multiply(matrix_pairs, num_threads=4):
    result_queue = queue.Queue()
    threads = []

    for idx, (A, B) in enumerate(matrix_pairs):
        t = threading.Thread(target=multiply_matrices, args=(A, B, result_queue, idx))
        threads.append(t)
        t.start()

        if len(threads) >= num_threads:
            for t in threads:
                t.join()
            threads = []

    for t in threads:
        t.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    return [r[1] for r in sorted(results)]


def make_matrix(rows, cols):
    import random

    return [[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]


pairs = [(make_matrix(50, 50), make_matrix(50, 50)) for _ in range(8)]
products = parallel_matrix_multiply(pairs)
