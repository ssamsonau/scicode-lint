import torch


def process_stream(model, stream, device="cuda", batch_size=32):
    model.to(device)
    model.eval()
    results = []
    batch = []
    for item in stream:
        batch.append(torch.tensor(item))
        if len(batch) >= batch_size:
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                output = model(batch_tensor)
            results.extend(output.cpu().numpy())
            batch = []
    if batch:
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            output = model(batch_tensor)
        results.extend(output.cpu().numpy())
    return results


class RealtimeClassifier:
    def __init__(self, model, batch_size=16):
        self.model = model.cuda().eval()
        self.batch_size = batch_size
        self.buffer = []

    def classify_batch(self, samples):
        x = torch.stack([torch.tensor(s) for s in samples]).cuda()
        with torch.no_grad():
            return self.model(x)
