import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))


def compute_embeddings(model, all_sequences):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    sequences_gpu = all_sequences.to(device)

    with torch.no_grad():
        embeddings = model(sequences_gpu)

    return embeddings.cpu().numpy()


def extract_features(encoder, dataset_tensor):
    encoder.cuda()
    encoder.eval()

    full_data = dataset_tensor.cuda()

    with torch.no_grad():
        features = encoder(full_data)

    return features.detach().cpu()
