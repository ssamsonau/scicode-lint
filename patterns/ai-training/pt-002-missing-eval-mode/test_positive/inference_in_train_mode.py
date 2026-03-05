import torch
import torch.nn as nn


class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


@torch.inference_mode()
def predict_batch(model, images):
    # BUG: Using inference_mode but never calling eval()
    # Dropout and batch norm still in training mode
    predictions = model(images)
    return predictions


def process_test_set(model, test_images):
    all_predictions = []
    # Processing without setting evaluation mode
    for img in test_images:
        pred = predict_batch(model, img.unsqueeze(0))
        all_predictions.append(pred.squeeze().cpu().numpy())
    return all_predictions
