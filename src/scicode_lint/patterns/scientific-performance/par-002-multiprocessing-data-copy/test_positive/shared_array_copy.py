from concurrent.futures import ProcessPoolExecutor

import numpy as np


class DataProcessor:
    def __init__(self, dataset):
        self.dataset = dataset

    def analyze(self, operation):
        if operation == "mean":
            return np.mean(self.dataset)
        elif operation == "std":
            return np.std(self.dataset)
        return np.sum(self.dataset)


def run_parallel_analysis(processor, operations):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(processor.analyze, op) for op in operations]
        return [f.result() for f in futures]


large_data = np.random.randn(50000, 500)
processor = DataProcessor(large_data)
results = run_parallel_analysis(processor, ["mean", "std", "sum", "mean"])
