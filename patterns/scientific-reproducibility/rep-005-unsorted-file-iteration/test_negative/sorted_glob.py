import glob
import os
from pathlib import Path


def load_sorted_files(directory):
    """Files loaded in sorted order - reproducible."""
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files]


def process_images_sorted(pattern):
    """Sorted glob for reproducible image processing order."""
    image_paths = sorted(glob.glob(pattern))
    images = []
    for path in image_paths:
        images.append(load_image(path))
    return images


def load_data_sorted(data_dir):
    """Using sorted() with Path.iterdir() for reproducibility."""
    data_path = Path(data_dir)
    return [f for f in sorted(data_path.iterdir()) if f.suffix == ".npy"]


def load_image(path):
    return path  # placeholder
