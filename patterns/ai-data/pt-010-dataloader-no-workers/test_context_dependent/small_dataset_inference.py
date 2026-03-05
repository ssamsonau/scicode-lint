import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SmallInferenceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx])


def run_inference_on_small_batch(model, test_samples):
    test_dataset = SmallInferenceDataset(test_samples)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            predictions.append(output.cpu().numpy())

    return np.vstack(predictions)


def quick_validation(model, val_data):
    val_dataset = SmallInferenceDataset(val_data)

    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs in val_loader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == inputs).sum().item()
            total_samples += inputs.size(0)

    accuracy = total_correct / total_samples
    return accuracy
