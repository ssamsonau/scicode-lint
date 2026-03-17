from contextlib import contextmanager

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, out_features)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@contextmanager
def inference_mode(model):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


def extract_features(model, image_batch):
    with inference_mode(model):
        features = model(image_batch)
    return features


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, inputs):
        with torch.no_grad():
            return self.model(inputs)

    def batch_predict(self, dataloader):
        results = []
        for batch in dataloader:
            results.append(self.predict(batch))
        return torch.cat(results)
