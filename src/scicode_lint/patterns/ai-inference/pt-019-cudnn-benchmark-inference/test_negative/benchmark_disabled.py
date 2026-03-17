import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


def detect_objects_multiscale(model, image_pyramid):
    torch.backends.cudnn.benchmark = False
    model.cuda()
    model.eval()

    detections = []
    with torch.no_grad():
        for scale_images in image_pyramid:
            preds = model(scale_images.cuda())
            detections.append(preds.cpu())
    return detections


def stream_inference(model, frame_source):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model.cuda()
    model.eval()

    with torch.no_grad():
        for frame in frame_source:
            yield model(frame.unsqueeze(0).cuda()).cpu()
