import glob
import json
import os
from pathlib import Path


def build_config_mapping(config_dir):
    config_map = {}
    for config_path in glob.glob(f"{config_dir}/*.json"):
        name = os.path.basename(config_path).replace(".json", "")
        with open(config_path) as f:
            config_map[name] = json.load(f)
    return config_map


def get_model_weights(weights_dir):
    weights = {}
    for weight_file in Path(weights_dir).glob("*.pt"):
        model_name = weight_file.stem
        weights[model_name] = weight_file
    return weights


def load_image_by_name(image_dir, name):
    image_paths = {os.path.basename(p): p for p in glob.glob(f"{image_dir}/*.png")}
    return image_paths.get(name)


def count_files(directory):
    return len(os.listdir(directory))


def find_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_*.pt")
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)
