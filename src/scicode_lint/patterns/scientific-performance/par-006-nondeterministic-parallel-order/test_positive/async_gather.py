import multiprocessing as mp

import numpy as np


def process_items(items, num_workers=4):
    def process(item):
        return np.sin(np.linspace(0, item, 128))

    with mp.Pool(num_workers) as pool:
        rows = list(pool.imap_unordered(process, items))
    spectrogram = np.array(rows)
    return spectrogram


def aggregate_async(data_chunks):
    results = []

    def callback(result):
        results.append(result)

    with mp.Pool() as pool:
        for chunk in data_chunks:
            pool.apply_async(np.fft.rfft, (chunk,), callback=callback)
        pool.close()
        pool.join()
    time_series = np.concatenate(results)
    return time_series
