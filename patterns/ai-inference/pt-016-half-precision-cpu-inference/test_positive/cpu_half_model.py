import torch


def run_cpu_inference(model, inputs):
    model.cpu()
    model.half()
    return model(inputs.half())
