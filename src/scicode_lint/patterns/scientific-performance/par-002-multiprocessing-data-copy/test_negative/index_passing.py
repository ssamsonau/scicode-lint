from multiprocessing import Pool

import numpy as np


def compute_chunk_stats(args):
    filepath, start, end = args
    data = np.load(filepath, mmap_mode="r")
    chunk = data[start:end]
    return {"mean": float(chunk.mean()), "std": float(chunk.std())}


def parallel_with_indices(filepath, total_rows, num_workers=4):
    chunk_size = total_rows // num_workers
    tasks = [
        (filepath, i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)
    ]

    with Pool(processes=num_workers) as pool:
        results = pool.map(compute_chunk_stats, tasks)

    return results
