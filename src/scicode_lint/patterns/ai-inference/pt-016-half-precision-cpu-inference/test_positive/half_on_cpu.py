import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc(x)


def run_cpu_inference(model, x):
    model = model.cpu().half()
    x = x.cpu().half()
    return model(x)


def deploy_model_cpu(checkpoint_path):
    model = SimpleModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.half()
    model.eval()
    return model


def process_batch_cpu(model, batch):
    model.to("cpu").to(torch.float16)
    results = []
    for item in batch:
        results.append(model(item.to(torch.float16)))
    return results
