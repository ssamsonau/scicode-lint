import pickle

import joblib


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def save_sklearn_model(model, path):
    joblib.dump(model, path)


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
