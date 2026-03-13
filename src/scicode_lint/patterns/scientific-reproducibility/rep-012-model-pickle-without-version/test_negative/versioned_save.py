import json
import pickle

import sklearn


def save_model_with_versions(model, model_path, metadata_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "sklearn_version": sklearn.__version__,
        "model_type": type(model).__name__,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def save_model_bundle(model, path):
    bundle = {
        "model": model,
        "sklearn_version": sklearn.__version__,
        "python_version": "3.10",
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def save_checkpoint_versioned(model, optimizer, epoch, path):
    import torch

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    torch.save(checkpoint, path)
