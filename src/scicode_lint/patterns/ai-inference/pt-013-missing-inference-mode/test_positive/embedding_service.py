import torch
import torch.nn as nn
import numpy as np


class SimilaritySearchEngine:
    def __init__(self, encoder, index_embeddings):
        self.encoder = encoder
        self.encoder.eval()
        self.device = torch.device("cuda")
        self.encoder.to(self.device)
        self.index = torch.from_numpy(index_embeddings).to(self.device)

    def find_similar(self, query_tokens, top_k=10):
        query = torch.tensor(query_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.encoder(query)
            query_norm = nn.functional.normalize(query_embedding, dim=-1)
            scores = torch.matmul(query_norm, self.index.T).squeeze(0)
            top_scores, top_indices = scores.topk(top_k)
        return top_indices.cpu().numpy(), top_scores.cpu().numpy()

    def encode_documents(self, token_batches):
        all_embeddings = []
        for tokens in token_batches:
            t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.encoder(t)
                all_embeddings.append(nn.functional.normalize(emb, dim=-1).cpu())
        return torch.cat(all_embeddings, dim=0).numpy()
