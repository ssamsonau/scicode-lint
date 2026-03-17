import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


class StreamingSpectrumDataset(IterableDataset):
    def __init__(self, file_paths, window_size=2048):
        self.file_paths = file_paths
        self.window_size = window_size

    def __iter__(self):
        for path in self.file_paths:
            signal = torch.load(path)
            for start in range(0, len(signal) - self.window_size, self.window_size // 2):
                chunk = signal[start : start + self.window_size]
                spectrum = torch.fft.rfft(chunk).abs()
                yield spectrum


class SpectralClassifier(nn.Module):
    def __init__(self, input_dim=1025, num_classes=6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def train_spectral_model(file_paths, epochs=20, lr=1e-3):
    dataset = StreamingSpectrumDataset(file_paths)
    loader = DataLoader(dataset, batch_size=256, num_workers=48, prefetch_factor=4)

    model = SpectralClassifier().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for spectra in loader:
            spectra = spectra.cuda()
            labels = torch.zeros(spectra.size(0), dtype=torch.long, device="cuda")
            loss = criterion(model(spectra), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
