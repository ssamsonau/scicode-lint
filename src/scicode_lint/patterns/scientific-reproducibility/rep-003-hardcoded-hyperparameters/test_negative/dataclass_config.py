from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    output_dim: int = 10


@dataclass
class TrainConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100


def create_model(input_dim, config: ModelConfig):
    layers = []
    prev_dim = input_dim
    for _ in range(config.num_layers):
        layers.extend(
            [nn.Linear(prev_dim, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)]
        )
        prev_dim = config.hidden_dim
    layers.append(nn.Linear(prev_dim, config.output_dim))
    return nn.Sequential(*layers)
