import glob
import os
from pathlib import Path


def load_all_configs(config_dir):
    configs = []
    for path in glob.glob(f"{config_dir}/*.yaml"):
        configs.append(path)
    return configs


def process_data_files(data_path):
    results = []
    for f in os.listdir(data_path):
        if f.endswith(".csv"):
            results.append(f)
    return results


def iterate_checkpoints(ckpt_dir):
    yield from Path(ckpt_dir).glob("*.pt")
