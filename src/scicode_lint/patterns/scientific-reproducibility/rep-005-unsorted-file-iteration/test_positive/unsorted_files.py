import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_image(path):
    return Image.open(path)


def load_dataset(data_dir):
    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            images.append(load_image(os.path.join(data_dir, filename)))
    return images


def process_all_csvs(pattern):
    dataframes = []
    for filepath in glob.glob(pattern):
        df = pd.read_csv(filepath)
        dataframes.append(df)
    return pd.concat(dataframes)


def iterate_data_files(data_path):
    path = Path(data_path)
    for file in path.iterdir():
        if file.suffix == ".npy":
            yield np.load(file)
