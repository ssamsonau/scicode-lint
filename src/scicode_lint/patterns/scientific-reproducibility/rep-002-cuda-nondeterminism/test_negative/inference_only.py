import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def run_inference(model, data, device):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        predictions = model(data)
        probabilities = torch.softmax(predictions, dim=1)

    return probabilities.cpu().numpy()


def batch_inference(model, dataloader, device):
    model.to(device)
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].cuda()
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())

    return torch.cat(all_predictions, dim=0)


def predict_single(model, sample):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        sample = sample.unsqueeze(0).to(device)
        output = model(sample)
        _, predicted = torch.max(output, 1)

    return predicted.item()
