import json
from pathlib import Path

import numpy as np


def load_from_manifest(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

    data = []
    base_dir = Path(manifest_path).parent
    for filename in manifest["files"]:
        filepath = base_dir / filename
        data.append(np.load(filepath))

    return np.concatenate(data)


def process_ordered_splits(config_path):
    with open(config_path) as f:
        config = json.load(f)

    results = []
    for split in config["splits"]:
        data = np.load(split["path"])
        results.append(data)
    return results


class DatasetLoader:
    def __init__(self, index_file):
        with open(index_file) as f:
            self.file_list = [line.strip() for line in f]

    def __iter__(self):
        for filepath in self.file_list:
            yield np.load(filepath)
