import multiprocessing as mp

import torch


def gpu_worker(rank, shared_config):
    torch.cuda.set_device(rank)
    model = torch.nn.Linear(shared_config["in_dim"], shared_config["out_dim"]).cuda()
    data = torch.randn(32, shared_config["in_dim"], device=f"cuda:{rank}")
    output = model(data)
    print(f"GPU worker {rank}: output shape {output.shape}")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    config = {"in_dim": 128, "out_dim": 10}

    processes = []
    for gpu_id in range(torch.cuda.device_count()):
        p = mp.Process(target=gpu_worker, args=(gpu_id, config))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
