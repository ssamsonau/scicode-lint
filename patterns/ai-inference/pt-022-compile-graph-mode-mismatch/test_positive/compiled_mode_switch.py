import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


def train_and_validate(model, train_loader, val_loader, optimizer, epochs):
    compiled_model = torch.compile(model)

    for epoch in range(epochs):
        compiled_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compiled_model(batch).mean()
            loss.backward()
            optimizer.step()

        compiled_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                _ = compiled_model(batch)


def full_pipeline(model, data):
    compiled = torch.compile(model, fullgraph=True)

    compiled.train()
    train_output = compiled(data)

    compiled.eval()
    eval_output = compiled(data)

    return train_output, eval_output
