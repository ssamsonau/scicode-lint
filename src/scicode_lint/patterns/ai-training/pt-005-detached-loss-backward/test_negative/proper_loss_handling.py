import torch.nn as nn


class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.task1_head = nn.Linear(64, 10)
        self.task2_head = nn.Linear(64, 5)

    def forward(self, x):
        shared_features = self.shared(x)
        return self.task1_head(shared_features), self.task2_head(shared_features)


def multi_task_train(model, data, labels1, labels2, optimizer):
    optimizer.zero_grad()

    out1, out2 = model(data)

    loss1 = nn.functional.cross_entropy(out1, labels1)
    loss2 = nn.functional.cross_entropy(out2, labels2)

    total_loss = loss1 + loss2

    total_loss.backward()
    optimizer.step()

    return loss1.item(), loss2.item(), total_loss.item()
