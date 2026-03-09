import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(self.embed(x).mean(dim=1))


model = EmbeddingModel(50000, 256)
model.eval()


def get_embedding(tokens):
    with torch.no_grad():
        return model(tokens)


def batch_embed(token_lists):
    embeddings = []
    for tokens in token_lists:
        with torch.no_grad():
            embeddings.append(model(tokens))
    return embeddings
