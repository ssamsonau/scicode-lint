import os

import numpy as np


def load_all_data(data_dir):
    all_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            filepath = os.path.join(data_dir, filename)
            data = np.load(filepath)
            all_data.append(data)

    return np.concatenate(all_data)


if __name__ == "__main__":
    data = load_all_data("./data")
    print(f"Loaded data shape: {data.shape}")
