def optimize_for_inference(model):
    model.eval()
    model.half()
    return model


def run_cpu_inference(model, inputs):
    model.cpu()
    model.half()
    return model(inputs.half())
