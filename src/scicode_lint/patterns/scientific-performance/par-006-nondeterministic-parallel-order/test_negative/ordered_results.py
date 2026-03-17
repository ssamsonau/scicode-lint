from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def fit_segment(indexed_segment):
    idx, segment = indexed_segment
    coeffs = np.polyfit(np.arange(len(segment)), segment, deg=2)
    return idx, coeffs


def parallel_piecewise_fit(signal, segment_size=256):
    segments = [
        (i, signal[i * segment_size : (i + 1) * segment_size])
        for i in range(len(signal) // segment_size)
    ]

    collected = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fit_segment, seg) for seg in segments]
        for future in as_completed(futures):
            idx, coeffs = future.result()
            collected[idx] = coeffs

    ordered_coeffs = [collected[k] for k in sorted(collected)]
    return np.array(ordered_coeffs)


recording = np.sin(np.linspace(0, 20 * np.pi, 4096)) + np.random.randn(4096) * 0.1
piecewise_coeffs = parallel_piecewise_fit(recording)
