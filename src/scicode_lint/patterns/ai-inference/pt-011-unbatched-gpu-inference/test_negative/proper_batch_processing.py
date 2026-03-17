import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TimeSeriesForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, forecast_steps):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, forecast_steps)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def forecast_batch(model, historical_data, batch_size=64):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    dataset = TensorDataset(historical_data)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    forecasts = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            pred = model(batch)
            forecasts.append(pred.cpu())

    return torch.cat(forecasts, dim=0)
