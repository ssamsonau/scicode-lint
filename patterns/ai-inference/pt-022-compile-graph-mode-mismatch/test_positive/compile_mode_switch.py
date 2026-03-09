import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.fc(self.dropout(x))


compiled_model = torch.compile(SimpleModel())


def train_and_eval(train_data, val_data):
    compiled_model.train()
    for batch in train_data:
        loss = compiled_model(batch).sum()
        loss.backward()

    compiled_model.eval()
    with torch.no_grad():
        for batch in val_data:
            compiled_model(batch)


def mixed_usage():
    model = torch.compile(SimpleModel())

    model.train()
    train_output = model(torch.randn(32, 512))

    model.eval()
    with torch.inference_mode():
        eval_output = model(torch.randn(32, 512))

    return train_output, eval_output
