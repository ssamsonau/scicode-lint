import numpy as np


def verify_computation(data):
    result = np.sum(data.astype(np.float32))
    expected = 42.5
    return result == expected


def compute_checksum(weights):
    checksum = np.zeros(1, dtype=np.float32)
    for w in weights:
        checksum += w.astype(np.float32).sum()
    return checksum[0]


def regression_test_output(model_output):
    reference = np.array([1.234, 5.678, 9.012], dtype=np.float32)
    return np.all(model_output == reference)


if __name__ == "__main__":
    data = np.random.randn(100)
    result = verify_computation(data)
    print(f"Verification: {result}")
