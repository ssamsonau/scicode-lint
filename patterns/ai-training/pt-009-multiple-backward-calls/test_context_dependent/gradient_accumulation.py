import torch
import torch.nn as nn
import torch.optim as optim


class LargeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src) + self.pos_encoder[:, : src.size(1), :]
        transformed = self.transformer(embedded)
        return self.fc(transformed)


def train_with_gradient_accumulation(model, data_loader, epochs, accumulation_steps=4):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        accumulated_loss = 0

        for batch_idx, (sequences, targets) in enumerate(data_loader):
            outputs = model(sequences)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            loss = loss / accumulation_steps
            loss.backward()

            accumulated_loss += loss

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulated_loss = 0

    return model
