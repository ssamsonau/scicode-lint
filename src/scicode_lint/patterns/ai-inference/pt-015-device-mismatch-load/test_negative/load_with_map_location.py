from pathlib import Path

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class ModelRegistry:
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = Path(model_dir)
        self.target_device = torch.device(device)
        self._cache = {}

    def get_model(self, name, model_class):
        if name in self._cache:
            return self._cache[name]

        path = self.model_dir / f"{name}.pt"
        weights = torch.load(path, map_location=self.target_device)
        model = model_class()
        model.load_state_dict(weights)
        model.to(self.target_device)
        model.eval()
        self._cache[name] = model
        return model

    def preload_all(self, model_configs):
        for name, config in model_configs.items():
            self.get_model(name, config["class"])
