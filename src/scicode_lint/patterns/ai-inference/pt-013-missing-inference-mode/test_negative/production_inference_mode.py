import torch
import torch.nn as nn


class SentimentAnalyzer:
    def __init__(self, model_path, vocab_size, embed_dim, hidden_dim, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(vocab_size, embed_dim, hidden_dim, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, vocab_size, embed_dim, hidden_dim, num_classes):
        return nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.LSTM(embed_dim, hidden_dim, batch_first=True),
        )

    @torch.inference_mode()
    def analyze(self, token_ids):
        tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        output = self.model(tokens)
        return torch.softmax(output, dim=-1).squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def analyze_batch(self, token_id_lists):
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(t, dtype=torch.long) for t in token_id_lists],
            batch_first=True,
        ).to(self.device)
        outputs = self.model(padded)
        return torch.softmax(outputs, dim=-1).cpu().numpy()
