import torch


def setup_inference_variable_shapes():
    """Benchmark disabled for variable-shape inference - correct."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_variable_input_inference(model, inputs):
    """Inference server handling variable input sizes without benchmark mode."""
    torch.backends.cudnn.benchmark = False
    model.eval()
    with torch.no_grad():
        return model(inputs)


class VariableSizePredictor:
    def __init__(self, model):
        torch.backends.cudnn.benchmark = False
        self.model = model.cuda()
        self.model.eval()

    def predict(self, batch):
        """Process batches of varying sizes without cudnn benchmark overhead."""
        with torch.no_grad():
            return self.model(batch.cuda())
