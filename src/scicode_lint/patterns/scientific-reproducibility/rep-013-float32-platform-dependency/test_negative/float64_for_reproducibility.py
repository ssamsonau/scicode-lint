import numpy as np


def compute_hash(data):
    result = np.sum(data.astype(np.float64))
    return result


def verify_computation(data, expected):
    result = np.sum(data.astype(np.float64))
    return result == expected


class ReproduciblePipeline:
    def __init__(self, data):
        self.data = data.astype(np.float64)

    def process(self):
        centered = self.data - self.data.mean()
        scaled = centered / centered.std()
        return scaled

    def checksum(self):
        return np.sum(self.data)

    def validate(self, expected_checksum):
        return self.checksum() == expected_checksum
