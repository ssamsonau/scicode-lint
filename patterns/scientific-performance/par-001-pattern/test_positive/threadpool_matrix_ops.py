from concurrent.futures import ThreadPoolExecutor

import numpy as np


def compute_eigenvalues(matrix):
    eigenvals, _ = np.linalg.eig(matrix)
    return np.sort(eigenvals)


def parallel_eigenvalue_analysis(matrices):
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(compute_eigenvalues, mat) for mat in matrices]
        for future in futures:
            results.append(future.result())
    return results


matrices = [np.random.randn(500, 500) for _ in range(20)]
eigenvalues = parallel_eigenvalue_analysis(matrices)
