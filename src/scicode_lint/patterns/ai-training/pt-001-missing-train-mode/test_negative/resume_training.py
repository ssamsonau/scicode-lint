import torch
import torch.nn as nn


class DistillationTrainer:
    def __init__(self, student, teacher, temperature=4.0):
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def train_epoch(self, dataloader):
        self.student.train()
        self.teacher.eval()

        epoch_loss = 0.0
        for inputs, labels in dataloader:
            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            student_logits = self.student(inputs)

            soft_targets = torch.softmax(teacher_logits / self.temperature, dim=-1)
            soft_preds = torch.log_softmax(student_logits / self.temperature, dim=-1)

            loss = self.kl_loss(soft_preds, soft_targets) * (self.temperature**2)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)
