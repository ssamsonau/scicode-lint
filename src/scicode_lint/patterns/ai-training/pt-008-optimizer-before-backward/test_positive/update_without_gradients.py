import torch.nn as nn
import torch.optim as optim


class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def train_rnn(model, sequences, labels, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        predictions = model(sequences)
        loss = loss_fn(predictions, labels)

        optimizer.step()

        # Then gradients cleared and computed
        model.zero_grad()
        loss.backward()

    return model
