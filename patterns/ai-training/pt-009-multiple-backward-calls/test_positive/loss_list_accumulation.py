import torch.nn as nn
import torch.optim as optim


class SequenceModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)


def train_sequence_model(model, train_data, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        loss_history = []

        for batch_sequences, batch_targets in train_data:
            optimizer.zero_grad()

            predictions = model(batch_sequences)
            loss = criterion(predictions.view(-1, predictions.size(-1)), batch_targets.view(-1))

            loss.backward()
            optimizer.step()

            # BUG: Storing tensor in list - retains computation graphs
            loss_history.append(loss)

        # Computing statistics from retained tensors
        _mean_loss = sum(loss_history) / len(loss_history)

    return model
