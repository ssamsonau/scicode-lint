"""Large scientific script: complete computer vision pipeline (~500 lines)."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Data Loading ──────────────────────────────────────────────────────────────


class ImageDataset(Dataset):
    """Custom dataset for loading image arrays with augmentation."""

    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        label = self.labels[idx]

        if self.augment:
            image = self._apply_augmentations(image)

        image = torch.FloatTensor(image)
        if image.dim() == 2:
            image = image.unsqueeze(0)

        return image, label

    def _apply_augmentations(self, image):
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, image.shape)
            image = image + noise
        if np.random.random() > 0.3:
            brightness = np.random.uniform(0.8, 1.2)
            image = image * brightness
        return np.clip(image, 0, 1).astype(np.float32)


def generate_synthetic_images(n_samples, img_size=32, n_channels=3, n_classes=10):
    """Generate synthetic image data for benchmarking."""
    images = np.random.randn(n_samples, n_channels, img_size, img_size).astype(np.float32)
    images = (images - images.min()) / (images.max() - images.min())

    labels = np.random.randint(0, n_classes, n_samples)
    class_probs = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02])
    labels = np.random.choice(n_classes, n_samples, p=class_probs)

    return images, labels


def create_weighted_sampler(labels):
    """Create a weighted sampler for imbalanced datasets."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(labels), replacement=True
    )
    return sampler


# ── Model Architecture ────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and residual connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SmallResNet(nn.Module):
    """Small ResNet-like model for image classification."""

    def __init__(self, n_classes=10, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = [ConvBlock(in_channels, out_channels, stride)]
        for _ in range(1, n_blocks):
            layers.append(ConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ── Training ──────────────────────────────────────────────────────────────────


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, epoch, **metrics):
        for name, value in metrics.items():
            self.history[name].append(value)

    def get_best(self, metric, mode="max"):
        values = self.history[metric]
        if mode == "max":
            idx = np.argmax(values)
        else:
            idx = np.argmin(values)
        return values[idx], idx

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(dict(self.history), f, indent=2)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    """Train model for one epoch with gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        probs = F.softmax(outputs, dim=1)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, val_f1, np.array(all_preds), np.array(all_targets)


def compute_detailed_metrics(y_true, y_pred, class_names=None):
    """Compute detailed classification metrics."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_acc.tolist(),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


# ── Cross-Validation ─────────────────────────────────────────────────────────


def run_cross_validation(images, labels, n_folds=5, n_epochs=30):
    """Run stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'=' * 50}")

        X_train, X_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_dataset = ImageDataset(X_train, y_train, augment=True)
        val_dataset = ImageDataset(X_val, y_val, augment=False)

        sampler = create_weighted_sampler(y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        n_classes = len(np.unique(labels))
        model = SmallResNet(n_classes=n_classes).to(device)

        class_counts = np.bincount(y_train, minlength=n_classes)
        class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6)).to(device)
        criterion = FocalLoss(alpha=class_weights)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        early_stopping = EarlyStopping(patience=7)
        tracker = MetricsTracker()

        best_val_f1 = 0.0
        best_model_state = None

        for epoch in range(n_epochs):
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
            scheduler.step()

            tracker.update(
                epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_f1=train_f1,
                val_loss=val_loss,
                val_acc=val_acc,
                val_f1=val_f1,
                lr=optimizer.param_groups[0]["lr"],
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            early_stopping.step(val_f1)
            if early_stopping.should_stop:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
            model = model.to(device)

        _, _, _, val_preds, val_targets = validate(model, val_loader, criterion, device)

        fold_metrics = compute_detailed_metrics(val_targets, val_preds)
        fold_metrics["best_val_f1"] = best_val_f1
        fold_results.append(fold_metrics)

        print(f"  Best Val F1: {best_val_f1:.4f}")

    mean_f1 = np.mean([r["best_val_f1"] for r in fold_results])
    std_f1 = np.std([r["best_val_f1"] for r in fold_results])
    print(f"\nCV Results: F1 = {mean_f1:.4f} ± {std_f1:.4f}")

    return fold_results


# ── Inference ─────────────────────────────────────────────────────────────────


def load_model_for_inference(checkpoint_path, n_classes=10, device=None):
    """Load a trained model for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallResNet(n_classes=n_classes)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model


def predict_batch(model, images, device, batch_size=128):
    """Run inference on a batch of images."""
    model.eval()
    all_preds = []
    all_probs = []

    dataset = ImageDataset(images, np.zeros(len(images)), augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_images, _ in loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def find_optimal_threshold(y_true, y_probs, target_class):
    """Find optimal classification threshold using precision-recall curve."""
    binary_true = (y_true == target_class).astype(int)
    class_probs = y_probs[:, target_class]

    precision, recall, thresholds = precision_recall_curve(binary_true, class_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)

    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    """Run the complete vision pipeline."""
    print("Generating synthetic dataset...")
    images, labels = generate_synthetic_images(5000)

    print(f"Dataset: {images.shape[0]} images, {len(np.unique(labels))} classes")
    print(f"Class distribution: {np.bincount(labels)}")

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print("\nRunning cross-validation on training set...")
    cv_results = run_cross_validation(X_train, y_train, n_folds=3, n_epochs=20)

    print("\nTraining final model on full training set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(np.unique(labels))
    model = SmallResNet(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    train_dataset = ImageDataset(X_train, y_train, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(20):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    torch.save(model.state_dict(), "final_model.pt")

    print("\nEvaluating on test set...")
    test_preds, test_probs = predict_batch(model, X_test, device)
    test_metrics = compute_detailed_metrics(y_test, test_preds)
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")

    results = {
        "cv_results": [
            {"best_val_f1": r["best_val_f1"], "macro_f1": r["macro_f1"]} for r in cv_results
        ],
        "test_metrics": {
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "per_class_accuracy": test_metrics["per_class_accuracy"],
        },
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
