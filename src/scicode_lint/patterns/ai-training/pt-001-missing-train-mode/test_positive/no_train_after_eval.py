import torch
import torch.nn as nn


class MetaLearner:
    def __init__(self, feature_extractor, classifier, inner_lr=0.01):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.meta_optimizer = torch.optim.Adam(
            list(feature_extractor.parameters()) + list(classifier.parameters()),
            lr=1e-3,
        )
        self.inner_lr = inner_lr

    def meta_train_step(self, support_set, query_set):
        self.meta_optimizer.zero_grad()

        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            support_features = self.feature_extractor(support_set[0])
            proto = support_features.mean(dim=0, keepdim=True)

        query_x, query_y = query_set
        query_features = self.feature_extractor(query_x)
        logits = self.classifier(query_features - proto)
        loss = nn.functional.cross_entropy(logits, query_y)
        loss.backward()
        self.meta_optimizer.step()
        return loss.item()


def training_loop(learner, episodes, n_epochs=100):
    losses = []
    for epoch in range(n_epochs):
        for support, query in episodes:
            loss = learner.meta_train_step(support, query)
            losses.append(loss)
    return losses
