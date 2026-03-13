import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))


def run_cpu_inference(model, inputs):
    model.cpu()
    model.eval()
    with torch.no_grad():
        return model(inputs.cpu())


class CPUInferenceServer:
    def __init__(self, model):
        self.model = model.cpu()
        self.model.eval()

    def predict(self, batch):
        with torch.inference_mode():
            return self.model(batch)

    def batch_predict(self, batches):
        results = []
        for batch in batches:
            pred = self.predict(batch)
            results.append(pred)
        return results
