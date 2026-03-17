import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] * 0.5 + torch.randn(1000) * 0.1).unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    epoch_loss = 0.0
    model.train()
    for batch_X, batch_y in loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch + 1}/10, Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    preds = model(X[:10])
    print("Sample predictions:", preds.squeeze().tolist())
