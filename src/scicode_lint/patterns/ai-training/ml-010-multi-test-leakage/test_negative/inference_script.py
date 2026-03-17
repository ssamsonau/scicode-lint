"""Inference script for making predictions."""

import pickle

import numpy as np


def load_model(model_path):
    """Load a pre-trained model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(model, X):
    """Make predictions on new data."""
    return model.predict(X)


def run_inference(model_path, data_path, output_path):
    """Run inference on new data."""
    model = load_model(model_path)

    X_new = np.load(data_path)

    predictions = predict(model, X_new)

    np.save(output_path, predictions)

    return predictions


if __name__ == "__main__":
    run_inference("model.pkl", "new_data.npy", "predictions.npy")
