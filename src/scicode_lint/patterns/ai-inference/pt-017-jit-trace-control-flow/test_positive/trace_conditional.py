import torch


def normalize_features(x):
    norms = x.norm(dim=-1, keepdim=True)
    has_zero = (norms == 0).any()
    if has_zero:
        norms = norms.clamp(min=1e-8)
    return x / norms


def remove_outliers(data, threshold=3.0):
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    z_scores = ((data - mean) / (std + 1e-8)).abs()
    if (z_scores > threshold).any():
        mask = (z_scores <= threshold).all(dim=1)
        return data[mask]
    return data


def export_preprocessing():
    example_features = torch.randn(32, 128)
    traced_norm = torch.jit.trace(normalize_features, example_features)

    example_data = torch.randn(100, 10)
    traced_outlier = torch.jit.trace(remove_outliers, example_data)

    return traced_norm, traced_outlier
