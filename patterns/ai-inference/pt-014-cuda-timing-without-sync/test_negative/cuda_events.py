import torch


def benchmark_with_events(model, inputs):
    """Uses CUDA events for accurate GPU timing - proper synchronization built-in."""
    model.eval()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        outputs = model(inputs)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    return outputs, elapsed_time


def profile_batch(model, batch, device):
    """CUDA events timing for batch profiling."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    batch = batch.to(device)
    start.record()
    result = model(batch)
    end.record()

    torch.cuda.synchronize()
    return result, start.elapsed_time(end)
