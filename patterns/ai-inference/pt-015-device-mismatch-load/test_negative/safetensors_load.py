from safetensors.torch import load_file


def load_model_safetensors(path, device):
    """Load model weights using safetensors format."""
    weights = load_file(path, device=device)
    return weights


def load_to_cpu(path):
    """Load safetensors file to CPU."""
    return load_file(path)


def load_checkpoint_safely(path, device):
    """Load safetensors checkpoint to specified device."""
    state_dict = load_file(path, device=str(device))
    return state_dict
