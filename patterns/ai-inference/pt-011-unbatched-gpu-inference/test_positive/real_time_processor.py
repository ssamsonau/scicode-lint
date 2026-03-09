import torch


def process_stream(model, stream, device="cuda"):
    model.to(device)
    model.eval()
    results = []
    for item in stream:
        tensor = torch.tensor(item).to(device)
        with torch.no_grad():
            output = model(tensor.unsqueeze(0))
        results.append(output.cpu().numpy())
    return results


class RealtimeClassifier:
    def __init__(self, model):
        self.model = model.cuda().eval()

    def classify(self, sample):
        x = torch.tensor(sample).cuda()
        with torch.no_grad():
            return self.model(x.unsqueeze(0))
