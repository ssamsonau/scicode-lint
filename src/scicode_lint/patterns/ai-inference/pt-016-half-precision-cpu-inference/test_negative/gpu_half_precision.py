import torch


def run_gpu_inference(model, inputs):
    model = model.cuda().half()
    inputs = inputs.cuda().half()
    with torch.no_grad():
        return model(inputs)


def fp16_inference_pipeline(model, data_loader, device="cuda"):
    """Half precision inference on GPU with tensor cores."""
    model = model.to(device, dtype=torch.float16)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device, dtype=torch.float16)
            results.append(model(batch))
    return results


class GPUInferenceEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.half()  # FP16 on GPU is fast

    def predict(self, inputs):
        inputs = inputs.to(self.device, dtype=torch.float16)
        with torch.no_grad():
            return self.model(inputs)
