import torch


class ModelServer:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.to("cuda")

    def predict(self, inputs):
        with torch.no_grad():
            return self.model(inputs)

    def batch_predict(self, batch):
        results = []
        with torch.no_grad():
            for item in batch:
                output = self.model(item.unsqueeze(0))
                results.append(output.squeeze())
        return torch.stack(results)
