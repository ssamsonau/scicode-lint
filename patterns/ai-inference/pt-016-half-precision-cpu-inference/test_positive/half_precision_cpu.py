import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)


def run_cpu_inference(model, input_data):
    model = model.half().cpu()
    model.eval()

    input_data = input_data.half().cpu()

    with torch.no_grad():
        output = model(input_data)
    return output


def deploy_model(model_path, device="cpu"):
    model = torch.load(model_path, map_location="cpu")
    model = model.cpu().half()
    model.eval()
    return model


def batch_inference_cpu(model, data_loader):
    model.eval()
    model = model.to(dtype=torch.float16, device="cpu")

    results = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(dtype=torch.float16)
            output = model(batch)
            results.append(output)

    return results


def optimize_for_deployment(model):
    model.eval()

    model = model.half()
    model = model.cpu()

    return model
