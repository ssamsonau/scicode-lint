import torch
from torch.utils.data import DataLoader, TensorDataset


def train_with_preloaded_data(raw_data, raw_labels, device):
    X = torch.tensor(raw_data, dtype=torch.float32).to(device)
    y = torch.tensor(raw_labels, dtype=torch.long).to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = torch.nn.Linear(X.shape[1], 10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()


def train_small_dataset_in_gpu_memory(features, targets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_gpu = features.cuda()
    y_gpu = targets.cuda()

    dataset = TensorDataset(X_gpu, y_gpu)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = torch.nn.Sequential(
        torch.nn.Linear(X_gpu.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
