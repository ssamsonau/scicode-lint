import torch


class BatchPredictor:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict_batch(self, inputs):
        with torch.no_grad():
            logits = self.model(inputs)
            predictions = (logits > self.threshold).int()
        return predictions


def classify_images(model, image_batch, conf_threshold=0.7):
    logits = model(image_batch)
    confident = logits > conf_threshold
    return confident
