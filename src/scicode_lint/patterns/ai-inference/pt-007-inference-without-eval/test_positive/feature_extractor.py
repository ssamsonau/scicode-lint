import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.base(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


def extract_features(model, data_loader):
    features = []
    with torch.no_grad():
        for batch in data_loader:
            feat = model(batch)
            features.append(feat.cpu())
    return torch.cat(features, dim=0)
