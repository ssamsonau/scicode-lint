import numpy as np


def encode_pixel_indices(height, width):
    row_idx = np.arange(height, dtype=np.int32)
    col_idx = np.arange(width, dtype=np.int32)
    return np.meshgrid(col_idx, row_idx)


def histogram_bin_counts(data, n_bins=256):
    bins = np.linspace(data.min(), data.max(), n_bins + 1)
    counts, _ = np.histogram(data, bins=bins)
    return counts.astype(np.int32)


def compute_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return np.array([row_min, col_min, row_max, col_max], dtype=np.int32)


def label_connected_components(binary_image):
    from scipy.ndimage import label
    labeled, n_features = label(binary_image)
    component_sizes = np.bincount(labeled.ravel())[1:]
    return labeled.astype(np.int32), component_sizes.astype(np.int32)


image = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
cols_grid, rows_grid = encode_pixel_indices(512, 512)
bin_counts = histogram_bin_counts(image.astype(float))
