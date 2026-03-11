import torch
from torch.utils.data import DataLoader, TensorDataset


def run_inference(model, data):
    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=16, num_workers=0)

    model.eval()
    predictions = []

    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch)
            predictions.append(output)

    return torch.cat(predictions)
