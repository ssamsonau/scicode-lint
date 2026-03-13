from safetensors.torch import load_file


def load_model_safetensors(path, device):
    weights = load_file(path, device=device)
    return weights


def load_to_cpu(path):
    return load_file(path)


def load_checkpoint_safely(path, device):
    state_dict = load_file(path, device=str(device))
    return state_dict
