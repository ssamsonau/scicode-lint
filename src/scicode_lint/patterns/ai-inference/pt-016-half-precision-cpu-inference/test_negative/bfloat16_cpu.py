import torch


def run_cpu_inference_bf16(model, inputs):
    """Run inference using bfloat16 precision."""
    model = model.cpu().to(dtype=torch.bfloat16)
    inputs = inputs.to(dtype=torch.bfloat16)
    with torch.no_grad():
        return model(inputs)


def bf16_batch_inference(model, batches):
    """Process multiple batches with bfloat16 precision."""
    model.eval()
    model = model.to(dtype=torch.bfloat16)
    results = []
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(dtype=torch.bfloat16)
            results.append(model(batch))
    return results


def quantized_cpu_inference(model, inputs):
    """Run quantized model inference on CPU."""
    model.cpu()
    model.eval()
    with torch.no_grad():
        return model(inputs)
