import torch


def run_quantized_cpu_inference(model, inputs):
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model(inputs)


def run_full_precision_cpu(model, inputs):
    model.eval()
    model.cpu()
    model.float()
    return model(inputs.float())


class CPUInferenceServer:
    def __init__(self, model_path):
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

    def predict(self, inputs):
        with torch.inference_mode():
            return self.model(inputs)
