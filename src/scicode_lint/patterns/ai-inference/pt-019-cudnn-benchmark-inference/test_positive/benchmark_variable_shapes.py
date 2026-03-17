import torch
import torch.nn as nn


class VideoFrameAnalyzer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return self.head(pooled)


def analyze_video_frames(model, frames_by_resolution):
    torch.backends.cudnn.benchmark = True
    model.cuda()
    model.eval()

    all_predictions = {}
    with torch.no_grad():
        for resolution, frames in frames_by_resolution.items():
            batch = frames.cuda()
            preds = model(batch)
            all_predictions[resolution] = preds.cpu()
    return all_predictions
