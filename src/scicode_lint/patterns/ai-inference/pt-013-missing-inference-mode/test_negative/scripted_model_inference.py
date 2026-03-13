import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def create_scripted_model(vocab_size, embed_dim, num_classes):
    model = TextClassifier(vocab_size, embed_dim, num_classes)
    model.eval()
    scripted = torch.jit.script(model)
    return scripted


@torch.inference_mode()
def run_scripted_inference(scripted_model, text_tensor, offsets):
    return scripted_model(text_tensor, offsets)


class InferenceServer:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text_tensor, offsets):
        return self.model(text_tensor, offsets)

    def batch_predict(self, batches):
        results = []
        with torch.inference_mode():
            for text, offsets in batches:
                output = self.model(text, offsets)
                results.append(output.argmax(dim=1))
        return results
