import torch
import torch.nn as nn


class PricePredictor(nn.Module):
    def __init__(self, num_features, hidden_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.layers(features)


def predict_house_prices(model, feature_data):
    model.eval()
    with torch.no_grad():
        prices = model(feature_data)
    return prices.squeeze()


def compute_prediction_errors(model, features, actual_prices):
    model.eval()
    with torch.no_grad():
        predicted = model(features).squeeze()
        errors = torch.abs(predicted - actual_prices)
        mean_error = errors.mean()
    return mean_error.item(), errors


def rank_predictions(model, data_batch):
    model.eval()
    with torch.no_grad():
        scores = model(data_batch).squeeze()
        sorted_indices = torch.argsort(scores, descending=True)
    return sorted_indices, scores[sorted_indices]
