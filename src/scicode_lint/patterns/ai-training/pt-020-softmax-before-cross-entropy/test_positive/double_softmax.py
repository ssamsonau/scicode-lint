import torch.nn.functional as F


def train_step(model, optimizer, batch, num_classes=100):
    images, labels = batch
    optimizer.zero_grad()

    features = model.backbone(images)
    logits = model.head(features)

    teacher_probs = F.softmax(model.teacher_head(features.detach()), dim=-1)
    student_probs = F.softmax(logits, dim=-1)
    distill_loss = F.cross_entropy(student_probs, teacher_probs.argmax(dim=-1))

    ce_loss = F.cross_entropy(F.softmax(logits, dim=-1), labels)
    loss = 0.5 * ce_loss + 0.5 * distill_loss

    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for batch in loader:
        total += train_step(model, optimizer, batch)
    return total / len(loader)
