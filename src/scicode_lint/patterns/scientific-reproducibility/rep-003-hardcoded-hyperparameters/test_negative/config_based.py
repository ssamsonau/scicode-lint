import argparse

import torch
import torch.nn as nn


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=784)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def build_model(config):
    model = nn.Sequential(
        nn.Linear(config.input_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout_rate),
        nn.Linear(config.hidden_dim, config.hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(config.hidden_dim // 2, config.output_dim),
    )
    return model


def train(model, config):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        batch = torch.randn(config.batch_size, config.input_dim)
        labels = torch.randint(0, config.output_dim, (config.batch_size,))

        optimizer.zero_grad()
        outputs = model(batch)
        loss = nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()


config = get_config()
model = build_model(config)
train(model, config)
