import glob
import os
from pathlib import Path

import numpy as np
import yaml


def load_all_configs(config_dir):
    """Load and merge configs - order affects override behavior."""
    merged_config = {}
    for path in glob.glob(f"{config_dir}/*.yaml"):
        with open(path) as f:
            config = yaml.safe_load(f)
            merged_config.update(config)
    return merged_config


def process_data_files(data_path):
    """Load and concatenate data files - order affects result."""
    arrays = []
    for f in os.listdir(data_path):
        if f.endswith(".npy"):
            arrays.append(np.load(os.path.join(data_path, f)))
    return np.concatenate(arrays)


def iterate_checkpoints(ckpt_dir):
    """Yield checkpoints - iteration order affects which is loaded first."""
    for ckpt_path in Path(ckpt_dir).glob("*.pt"):
        yield ckpt_path
