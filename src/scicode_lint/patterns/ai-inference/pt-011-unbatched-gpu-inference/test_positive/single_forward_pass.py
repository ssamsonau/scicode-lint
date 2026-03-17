import torch
import torch.nn as nn


class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def compute_anomaly_scores(model, sensor_readings):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    all_data = sensor_readings.to(device)

    with torch.no_grad():
        reconstructed = model(all_data)
        scores = ((all_data - reconstructed) ** 2).mean(dim=1)

    return scores.cpu().numpy()


def detect_anomalies(autoencoder, full_dataset, threshold):
    autoencoder.cuda()
    autoencoder.eval()

    data_gpu = full_dataset.cuda()

    with torch.no_grad():
        reconstruction = autoencoder(data_gpu)
        mse = ((data_gpu - reconstruction) ** 2).mean(dim=1)
        anomalies = mse > threshold

    return anomalies.cpu()
