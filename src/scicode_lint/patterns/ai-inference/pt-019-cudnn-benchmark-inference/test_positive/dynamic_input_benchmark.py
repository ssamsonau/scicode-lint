import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def classify_texts(model, tokenized_texts):
    cudnn.benchmark = True
    model.eval()
    model.cuda()

    results = []
    with torch.no_grad():
        for tokens in tokenized_texts:
            tensor = torch.tensor(tokens).unsqueeze(0).cuda()
            output = model(tensor)
            results.append(output.argmax(dim=-1).item())
    return results
