import torch


def train_and_eval(model, train_data, test_data):
    compiled = torch.compile(model)

    compiled.train()
    for x, y in train_data:
        loss = compiled(x).sum()
        loss.backward()

    compiled.eval()
    with torch.no_grad():
        for x, _ in test_data:
            _ = compiled(x)
