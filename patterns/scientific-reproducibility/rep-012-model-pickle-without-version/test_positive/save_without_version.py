import pickle

import joblib


def save_model(model, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def save_sklearn_model(model, path):
    joblib.dump(model, path)


def checkpoint_model(model, optimizer, epoch, path):
    checkpoint = {"epoch": epoch, "model": model, "optimizer_state": optimizer.state_dict()}
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
