import torch
import torch.nn as nn


class SpeechEncoder(nn.Module):
    def __init__(self, n_mels, hidden_dim, n_classes):
        super().__init__()
        self.conv = nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, mel_spec):
        x = torch.relu(self.conv(mel_spec))
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        return self.head(h.squeeze(0))


def transcribe_on_cpu(model, audio_features):
    model.cpu()
    model.half()
    model.eval()

    features = audio_features.cpu().half()
    with torch.no_grad():
        logits = model(features)
    return logits.argmax(dim=-1)
