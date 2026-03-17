import torch
import torch.nn as nn


class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def predict_category(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        predictions = logits.argmax(dim=1)
    return predictions


def get_top_k_predictions(model, inputs, k=3):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k, dim=1)
    return top_indices, top_probs
