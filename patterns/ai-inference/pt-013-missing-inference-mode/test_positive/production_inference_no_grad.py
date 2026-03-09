import torch
import torch.nn as nn


class ProductionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc(x)


model = ProductionModel()
model.eval()


def serve_prediction(input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        return output.argmax(dim=1)


def batch_inference(batch):
    results = []
    with torch.no_grad():
        for item in batch:
            pred = model(item)
            results.append(pred)
    return results
