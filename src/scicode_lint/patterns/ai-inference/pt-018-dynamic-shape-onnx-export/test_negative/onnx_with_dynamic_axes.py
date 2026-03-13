import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output[:, -1, :])


def export_model_for_serving(model, save_path):
    model.eval()
    batch_size = 1
    seq_length = 32
    dummy_input = torch.randint(0, 1000, (batch_size, seq_length))

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size"},
        },
        opset_version=14,
    )
