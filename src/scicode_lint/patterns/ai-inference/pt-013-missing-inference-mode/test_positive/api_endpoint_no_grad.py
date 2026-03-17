import torch
import torch.nn as nn
from torchvision import transforms


class ImageClassificationService:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify_image(self, image_tensor):
        processed = self.preprocess(image_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(processed)
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze().cpu()

    def classify_batch(self, image_tensors):
        batch = torch.stack([self.preprocess(img) for img in image_tensors]).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
        return torch.softmax(logits, dim=-1).cpu()
