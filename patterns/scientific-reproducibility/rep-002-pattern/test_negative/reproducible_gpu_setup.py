import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def create_transformer():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
    )

    transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
    transformer = transformer.to(device)

    return transformer, device


model, dev = create_transformer()
input_seq = torch.randn(4, 10, 512).to(dev)
encoded = model(input_seq)
