import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class InferenceService:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        self.model = Classifier(256, 10)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            logits = self.model(inputs)
            return logits.argmax(dim=1)

    def predict_batch(self, batch_inputs):
        results = []
        for inputs in batch_inputs:
            pred = self.predict(inputs)
            results.append(pred)
        return results
