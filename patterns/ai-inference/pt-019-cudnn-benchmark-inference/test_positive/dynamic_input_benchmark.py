import torch.backends.cudnn as cudnn


def setup_inference():
    cudnn.benchmark = True


def serve_requests(model, requests):
    cudnn.benchmark = True
    results = []
    for req in requests:
        results.append(model(req))
    return results
