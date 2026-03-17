import torch
import torch.nn as nn


class TextModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embed(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(hidden.squeeze(0))


def online_learning_loop(model, data_stream, optimizer):
    compiled_model = torch.compile(model)

    for batch in data_stream:
        if batch.get("is_labeled"):
            compiled_model.train()
            optimizer.zero_grad()
            logits = compiled_model(batch["tokens"])
            loss = nn.functional.cross_entropy(logits, batch["labels"])
            loss.backward()
            optimizer.step()

        compiled_model.eval()
        with torch.no_grad():
            predictions = compiled_model(batch["tokens"])
            yield torch.softmax(predictions, dim=-1)
