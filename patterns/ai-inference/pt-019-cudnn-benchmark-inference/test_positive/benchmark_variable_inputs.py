import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def inference_server(model, request_queue):
    model.eval()
    model.cuda()

    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        while True:
            batch = request_queue.get()
            if batch is None:
                break

            images = batch["images"].cuda()
            embeddings = model(images)
            yield embeddings.cpu()


def process_images(model, images_list):
    model.eval()
    model.cuda()

    torch.backends.cudnn.benchmark = True

    results = []
    with torch.no_grad():
        for images in images_list:
            output = model(images.cuda())
            results.append(output.cpu())
    return results
