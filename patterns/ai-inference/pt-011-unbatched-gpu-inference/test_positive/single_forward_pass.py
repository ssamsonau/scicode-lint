import torch
import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test_model(model, X_test, y_test):
    if torch.cuda.is_available():
        model.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        predictions = logits.argmax(dim=1)
        correct = (predictions == y_test).sum().item()
        accuracy = correct / len(y_test)

    return accuracy


def predict_all(model, test_data):
    device = torch.device("cuda:0")
    model.to(device)
    test_tensor = torch.from_numpy(test_data).float().to(device)

    model.eval()
    with torch.no_grad():
        all_predictions = model(test_tensor)

    return all_predictions.cpu()
