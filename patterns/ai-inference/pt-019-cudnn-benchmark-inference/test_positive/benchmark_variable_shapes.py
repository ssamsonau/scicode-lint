import torch
import torch.backends.cudnn as cudnn


def setup_inference():
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True


def run_variable_batch_inference(model, images):
    torch.backends.cudnn.benchmark = True
    model.cuda()
    model.eval()

    results = []
    for batch in images:
        with torch.no_grad():
            output = model(batch.cuda())
            results.append(output)
    return results


def serve_requests(model, request_queue, response_queue):
    cudnn.benchmark = True
    while True:
        request = request_queue.get()
        result = model(request)
        response_queue.put(result)
