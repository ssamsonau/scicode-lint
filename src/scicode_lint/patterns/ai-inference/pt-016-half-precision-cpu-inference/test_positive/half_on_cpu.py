import torch
import torch.nn as nn
from pathlib import Path


class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.fc = nn.Linear(embed_dim * 2, 1)

    def forward(self, user_ids, item_ids):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        return self.fc(torch.cat([u, i], dim=-1)).squeeze(-1)


def deploy_recommender(checkpoint_path):
    model = torch.load(checkpoint_path, map_location="cpu")
    model.half()
    model.eval()
    return model


def score_candidates(model, user_id, candidate_items):
    model.cpu().to(torch.float16)
    users = torch.full((len(candidate_items),), user_id, dtype=torch.long)
    items = torch.tensor(candidate_items, dtype=torch.long)
    with torch.no_grad():
        scores = model(users, items)
    return scores
