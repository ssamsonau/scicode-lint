import numpy as np


def make_layer(input_dim, output_dim, weight_cache={}):
    key = (input_dim, output_dim)
    if key not in weight_cache:
        weight_cache[key] = np.random.randn(input_dim, output_dim) * 0.01
    return weight_cache[key]


def log_metric(epoch, loss, log_entries=[]):
    log_entries.append({"epoch": epoch, "loss": loss})
    if len(log_entries) % 10 == 0:
        avg = np.mean([e["loss"] for e in log_entries[-10:]])
        return avg
    return loss


class FeatureSelector:
    def __init__(self, selected_features=set()):
        self.features = selected_features

    def add(self, name, importance):
        if importance > 0.5:
            self.features.add(name)

    def get_mask(self, all_features):
        return [f in self.features for f in all_features]
