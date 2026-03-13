import numpy as np


def sort_unique_ids(user_ids):
    assert len(user_ids) == len(np.unique(user_ids)), "IDs must be unique"
    return np.sort(user_ids)


def rank_by_unique_key(items, unique_keys):
    assert len(unique_keys) == len(np.unique(unique_keys)), "Keys must be unique"
    order = np.argsort(unique_keys)
    return items[order]


def get_sorted_timestamps(timestamps):
    if len(timestamps) != len(np.unique(timestamps)):
        raise ValueError("Timestamps must be unique")
    return np.sort(timestamps)


class UniqueKeySorter:
    def __init__(self, data, key_column):
        self.data = data
        self.key_column = key_column

    def sort(self):
        keys = self.data[self.key_column]
        if len(keys) != len(np.unique(keys)):
            raise ValueError("Key column must have unique values")
        order = np.argsort(keys)
        return {k: v[order] for k, v in self.data.items()}
