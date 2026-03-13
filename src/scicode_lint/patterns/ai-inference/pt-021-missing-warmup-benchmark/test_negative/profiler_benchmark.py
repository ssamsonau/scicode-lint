import torch
from torch.profiler import ProfilerActivity, profile


def profile_model_inference(model, input_tensor):
    model.cuda()
    model.eval()
    input_tensor = input_tensor.cuda()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(50):
                model(input_tensor)

    return prof.key_averages()


def detailed_profile(model, inputs, wait=2, warmup=3, active=5):
    model.cuda()
    model.eval()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
    ) as prof:
        for step in range(wait + warmup + active):
            model(inputs.cuda())
            prof.step()

    return prof
