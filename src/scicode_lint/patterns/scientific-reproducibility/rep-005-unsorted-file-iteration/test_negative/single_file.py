import numpy as np
import pandas as pd


def load_single_dataset(filepath):
    return np.load(filepath)


def load_parquet_dataset(filepath):
    return pd.read_parquet(filepath)


def process_hdf5_dataset(filepath):
    import h5py

    with h5py.File(filepath, "r") as f:
        data = f["data"][:]
        labels = f["labels"][:]
    return data, labels


class SingleFileDataset:
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
