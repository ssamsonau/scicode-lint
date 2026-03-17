import torch
import torch.nn as nn


class SegmentationNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


def evaluate_segmentation(model, image_paths, preprocess_fn, batch_size=8):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    all_masks = []
    batch = []

    with torch.no_grad():
        for path in image_paths:
            batch.append(preprocess_fn(path))
            if len(batch) == batch_size:
                tensor = torch.stack(batch).to(device)
                masks = model(tensor).argmax(dim=1)
                all_masks.append(masks.cpu())
                batch = []

        if batch:
            tensor = torch.stack(batch).to(device)
            masks = model(tensor).argmax(dim=1)
            all_masks.append(masks.cpu())

    return torch.cat(all_masks, dim=0)
