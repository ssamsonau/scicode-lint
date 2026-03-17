import numpy as np


class DataPipeline:
    def __init__(self, data):
        self.data = data
        self.row_means = data.mean(axis=1)
        self.row_stds = data.std(axis=1)

    def center(self):
        return self.data - self.row_means

    def standardize(self):
        centered = self.data - self.row_means
        return centered / self.row_stds

    def clip_outliers(self, threshold=2.0):
        z = (self.data - self.row_means) / self.row_stds
        return np.clip(z, -threshold, threshold)
