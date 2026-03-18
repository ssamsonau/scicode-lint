"""Extra-large scientific script: full experiment framework (~1000 lines).

Includes: data management, model zoo, training, hyperparameter search,
evaluation, calibration, ensembling, and experiment tracking.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataConfig:
    """Configuration for data generation and loading."""

    n_samples: int = 50000
    n_features: int = 100
    n_classes: int = 10
    test_size: float = 0.15
    val_size: float = 0.10
    imbalance_ratio: float = 10.0
    noise_level: float = 0.1
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_type: str = "resnet"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    activation: str = "relu"
    weight_init: str = "kaiming"


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    n_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 15
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    use_amp: bool = False


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    n_folds: int = 5
    ensemble_size: int = 3
    output_dir: str = "experiments"

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dict for serialization."""
        return {
            "data": self.data.__dict__,
            "model": {**self.model.__dict__},
            "training": self.training.__dict__,
            "n_folds": self.n_folds,
            "ensemble_size": self.ensemble_size,
        }

    def fingerprint(self) -> str:
        """Deterministic hash of config for caching."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# Data Management
# ══════════════════════════════════════════════════════════════════════════════


class TabularDataset(Dataset):
    """Dataset for tabular data with optional feature scaling."""

    def __init__(self, features, labels, scaler=None, fit_scaler=False):
        if fit_scaler and scaler is not None:
            self.features = scaler.fit_transform(features)
        elif scaler is not None:
            self.features = scaler.transform(features)
        else:
            self.features = features.copy()

        self.labels = labels.copy()
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_imbalanced_data(config: DataConfig):
    """Generate synthetic imbalanced dataset."""
    rng = np.random.RandomState(config.seed)

    # Generate class-specific data
    n_per_class = []
    base_count = config.n_samples // config.n_classes
    for i in range(config.n_classes):
        scale = config.imbalance_ratio ** (i / (config.n_classes - 1))
        n_per_class.append(max(int(base_count / scale), 10))

    all_features = []
    all_labels = []

    for class_idx, n in enumerate(n_per_class):
        center = rng.randn(config.n_features) * 2
        spread = rng.uniform(0.5, 2.0, config.n_features)
        features = rng.randn(n, config.n_features) * spread + center
        features += rng.randn(n, config.n_features) * config.noise_level

        all_features.append(features)
        all_labels.extend([class_idx] * n)

    X = np.vstack(all_features).astype(np.float32)
    y = np.array(all_labels)

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


def create_data_splits(X, y, config: DataConfig):
    """Create train/val/test splits with stratification."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )

    val_fraction = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=config.seed,
    )

    logger.info(f"Data splits: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def create_dataloaders(splits, config, scaler=None):
    """Create DataLoaders with class-weighted sampling for training."""
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    train_dataset = TabularDataset(X_train, y_train, scaler=scaler, fit_scaler=True)
    val_dataset = TabularDataset(X_val, y_val, scaler=scaler, fit_scaler=False)

    # Weighted sampling for imbalanced data
    class_counts = np.bincount(y_train, minlength=config.data.n_classes)
    sample_weights = 1.0 / class_counts[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size * 2, shuffle=False)

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# Model Zoo
# ══════════════════════════════════════════════════════════════════════════════


def get_activation(name):
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
    }
    return activations[name]()


class MLPClassifier(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(self, input_dim, config: ModelConfig):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 10)

        self._init_weights(config.weight_init)

    def _init_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class ResidualBlock(nn.Module):
    """Residual block for tabular data."""

    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class ResNetTabular(nn.Module):
    """ResNet-style model for tabular data."""

    def __init__(self, input_dim, config: ModelConfig):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dims[0]),
            nn.BatchNorm1d(config.hidden_dims[0]),
            nn.ReLU(),
        )

        blocks = []
        for dim in config.hidden_dims:
            blocks.append(ResidualBlock(dim, config.dropout_rate))

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(config.hidden_dims[-1], 10)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.classifier(x)


def create_model(input_dim, config: ModelConfig):
    """Factory function to create models by type."""
    models = {
        "mlp": MLPClassifier,
        "resnet": ResNetTabular,
    }

    model_cls = models.get(config.model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model_cls(input_dim, config)


# ══════════════════════════════════════════════════════════════════════════════
# Training Engine
# ══════════════════════════════════════════════════════════════════════════════


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(base_lr * factor, self.min_lr)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    """Handles the training loop with all bells and whistles."""

    def __init__(self, model, config: TrainingConfig, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer, config.warmup_epochs, config.n_epochs
        )

        self.scaler = torch.amp.GradScaler(enabled=config.use_amp)
        self.metrics = MetricsLogger()
        self.best_val_metric = 0.0
        self.best_model_state = None
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if self.config.mixup_alpha > 0:
                inputs, targets_a, targets_b, lam = self._mixup(inputs, targets)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.config.use_amp):
                outputs = self.model(inputs)

                if self.config.mixup_alpha > 0:
                    loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(
                        outputs, targets_b
                    )
                else:
                    loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return running_loss / total, correct / total

    def _mixup(self, inputs, targets):
        """Apply mixup augmentation."""
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        idx = torch.randperm(inputs.size(0), device=self.device)
        mixed = lam * inputs + (1 - lam) * inputs[idx]
        return mixed, targets, targets[idx], lam

    @torch.no_grad()
    def validate(self, val_loader):
        """Run validation."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        total = len(all_targets)
        val_loss = running_loss / total
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return val_loss, val_acc, val_f1, np.array(all_probs), np.array(all_targets)

    def fit(self, train_loader, val_loader):
        """Run full training loop."""
        for epoch in range(self.config.n_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, _, _ = self.validate(val_loader)
            self.scheduler.step()

            self.metrics.log(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_f1=val_f1,
                lr=self.scheduler.get_lr(),
            )

            if val_f1 > self.best_val_metric:
                self.best_val_metric = val_f1
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                    f"val_f1={val_f1:.4f}, lr={self.scheduler.get_lr():.6f}"
                )

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model = self.model.to(self.device)

        return self.metrics


class MetricsLogger:
    """Log and export training metrics."""

    def __init__(self):
        self.records = []

    def log(self, **kwargs):
        self.records.append(kwargs)

    def to_dict(self):
        return self.records

    def get_best(self, metric, mode="max"):
        values = [r[metric] for r in self.records]
        idx = np.argmax(values) if mode == "max" else np.argmin(values)
        return values[idx], idx


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation & Calibration
# ══════════════════════════════════════════════════════════════════════════════


def comprehensive_evaluation(y_true, y_probs, class_names=None):
    """Compute comprehensive evaluation metrics."""
    y_pred = np.argmax(y_probs, axis=1)
    n_classes = y_probs.shape[1]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "log_loss": log_loss(y_true, y_probs, labels=list(range(n_classes))),
    }

    # Per-class metrics
    try:
        if n_classes == 2:
            metrics["auroc"] = roc_auc_score(y_true, y_probs[:, 1])
            metrics["avg_precision"] = average_precision_score(y_true, y_probs[:, 1])
        else:
            y_true_onehot = np.eye(n_classes)[y_true]
            metrics["auroc_macro"] = roc_auc_score(
                y_true_onehot, y_probs, average="macro", multi_class="ovr"
            )
    except ValueError:
        pass

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = np.zeros(n_classes)
    for i in range(n_classes):
        if cm[i].sum() > 0:
            per_class_acc[i] = cm[i, i] / cm[i].sum()
    metrics["per_class_accuracy"] = per_class_acc.tolist()
    metrics["confusion_matrix"] = cm.tolist()

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["classification_report"] = report

    return metrics


def compute_calibration(y_true, y_probs, n_bins=15):
    """Compute calibration metrics and reliability diagram data."""
    y_pred = np.argmax(y_probs, axis=1)
    n_classes = y_probs.shape[1]

    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Expected Calibration Error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability_data = []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)
            reliability_data.append(
                {
                    "bin_center": (bin_boundaries[i] + bin_boundaries[i + 1]) / 2,
                    "accuracy": float(bin_acc),
                    "confidence": float(bin_conf),
                    "count": int(bin_size),
                }
            )

    # Per-class calibration
    per_class_cal = {}
    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
            try:
                prob_true, prob_pred = calibration_curve(
                    binary_true, y_probs[:, c], n_bins=min(n_bins, 10)
                )
                per_class_cal[c] = {
                    "prob_true": prob_true.tolist(),
                    "prob_pred": prob_pred.tolist(),
                }
            except ValueError:
                pass

    return {
        "ece": float(ece),
        "reliability_diagram": reliability_data,
        "per_class_calibration": per_class_cal,
        "mean_confidence": float(confidences.mean()),
        "accuracy": float(accuracies.mean()),
    }


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """Optimize temperature on validation set."""
        logits_tensor = torch.FloatTensor(logits)
        labels_tensor = torch.LongTensor(labels)

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits_tensor)
            loss = criterion(scaled, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        logger.info(f"Learned temperature: {self.temperature.item():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Ensembling
# ══════════════════════════════════════════════════════════════════════════════


class ModelEnsemble:
    """Ensemble of trained models with various combination strategies."""

    def __init__(self):
        self.models = []
        self.weights = []

    def add_model(self, model, weight=1.0):
        self.models.append(model)
        self.weights.append(weight)

    @torch.no_grad()
    def predict(self, dataloader, device, strategy="soft_vote"):
        """Generate ensemble predictions."""
        all_probs = []

        for model in self.models:
            model.eval()
            model_probs = []

            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                model_probs.append(probs.cpu().numpy())

            all_probs.append(np.vstack(model_probs))

        if strategy == "soft_vote":
            weights = np.array(self.weights) / sum(self.weights)
            ensemble_probs = sum(w * p for w, p in zip(weights, all_probs))
        elif strategy == "hard_vote":
            preds = [np.argmax(p, axis=1) for p in all_probs]
            preds_stack = np.stack(preds, axis=0)
            ensemble_preds = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=all_probs[0].shape[1]).argmax(),
                axis=0,
                arr=preds_stack,
            )
            ensemble_probs = np.eye(all_probs[0].shape[1])[ensemble_preds]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return ensemble_probs

    def diversity_metrics(self, dataloader, device):
        """Compute pairwise disagreement between models."""
        all_preds = []

        for model in self.models:
            model.eval()
            preds = []
            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds.extend(outputs.argmax(1).cpu().numpy())
            all_preds.append(np.array(preds))

        n_models = len(all_preds)
        disagreements = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagree = (all_preds[i] != all_preds[j]).mean()
                disagreements[i, j] = disagree
                disagreements[j, i] = disagree

        return {
            "pairwise_disagreement": disagreements.tolist(),
            "mean_disagreement": float(disagreements[np.triu_indices(n_models, k=1)].mean()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Experiment Runner
# ══════════════════════════════════════════════════════════════════════════════


class ExperimentRunner:
    """Orchestrates the full experiment pipeline."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output_dir) / config.fingerprint()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Run the full experiment."""
        start_time = time.time()

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Generate data
        logger.info("Generating data...")
        X, y = generate_imbalanced_data(self.config.data)
        splits = create_data_splits(X, y, self.config.data)

        # Cross-validation
        logger.info("Running cross-validation...")
        cv_results = self._run_cv(splits["train"][0], splits["train"][1])

        # Train ensemble
        logger.info("Training ensemble...")
        ensemble, scaler = self._train_ensemble(splits)

        # Evaluate
        logger.info("Evaluating...")
        eval_results = self._evaluate_ensemble(ensemble, splits, scaler)

        # Calibration
        logger.info("Calibrating...")
        cal_results = self._calibrate(ensemble, splits, scaler)

        total_time = time.time() - start_time

        results = {
            "config": self.config.to_dict(),
            "cv_results": cv_results,
            "eval_results": eval_results,
            "calibration": cal_results,
            "total_time_seconds": total_time,
            "device": str(self.device),
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Experiment completed in {total_time:.1f}s")
        return results

    def _run_cv(self, X, y):
        """Run stratified k-fold cross-validation."""
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.data.seed,
        )

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            train_dataset = TabularDataset(X_train_scaled, y_train)
            val_dataset = TabularDataset(X_val_scaled, y_val)

            train_loader = DataLoader(
                train_dataset, batch_size=self.config.training.batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size * 2)

            model = create_model(X.shape[1], self.config.model)
            trainer = Trainer(model, self.config.training, self.device)
            trainer.fit(train_loader, val_loader)

            _, _, val_f1, val_probs, val_targets = trainer.validate(val_loader)

            fold_result = comprehensive_evaluation(val_targets, val_probs)
            fold_result["fold"] = fold
            fold_metrics.append(fold_result)

            logger.info(f"Fold {fold + 1}: F1={val_f1:.4f}")

        mean_f1 = np.mean([m["macro_f1"] for m in fold_metrics])
        std_f1 = np.std([m["macro_f1"] for m in fold_metrics])

        return {
            "folds": fold_metrics,
            "mean_f1": float(mean_f1),
            "std_f1": float(std_f1),
        }

    def _train_ensemble(self, splits):
        """Train an ensemble of models."""
        X_train, y_train = splits["train"]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        ensemble = ModelEnsemble()

        for i in range(self.config.ensemble_size):
            logger.info(f"Training ensemble member {i + 1}/{self.config.ensemble_size}")

            # Bootstrap sample
            rng = np.random.RandomState(self.config.data.seed + i)
            boot_idx = rng.choice(len(y_train), len(y_train), replace=True)
            oob_idx = np.setdiff1d(np.arange(len(y_train)), boot_idx)

            train_dataset = TabularDataset(X_train_scaled[boot_idx], y_train[boot_idx])
            val_dataset = TabularDataset(X_train_scaled[oob_idx], y_train[oob_idx])

            train_loader = DataLoader(
                train_dataset, batch_size=self.config.training.batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size * 2)

            model = create_model(X_train.shape[1], self.config.model)
            trainer = Trainer(model, self.config.training, self.device)
            trainer.fit(train_loader, val_loader)

            ensemble.add_model(trainer.model)

        return ensemble, scaler

    def _evaluate_ensemble(self, ensemble, splits, scaler):
        """Evaluate ensemble on test set."""
        X_test, y_test = splits["test"]
        X_test_scaled = scaler.transform(X_test)

        test_dataset = TabularDataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size * 2)

        ensemble_probs = ensemble.predict(test_loader, self.device, strategy="soft_vote")
        metrics = comprehensive_evaluation(y_test, ensemble_probs)

        diversity = ensemble.diversity_metrics(test_loader, self.device)
        metrics["ensemble_diversity"] = diversity

        # Compare with individual models
        individual_f1s = []
        for i, model in enumerate(ensemble.models):
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    preds = model(inputs).argmax(1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(targets.numpy())

            f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
            individual_f1s.append(f1)

        metrics["individual_f1s"] = individual_f1s
        metrics["ensemble_gain"] = float(metrics["macro_f1"] - np.mean(individual_f1s))

        logger.info(
            f"Ensemble F1: {metrics['macro_f1']:.4f} "
            f"(+{metrics['ensemble_gain']:.4f} over avg individual)"
        )

        return metrics

    def _calibrate(self, ensemble, splits, scaler):
        """Calibrate ensemble predictions with temperature scaling."""
        X_val, y_val = splits["val"]
        X_val_scaled = scaler.transform(X_val)

        val_dataset = TabularDataset(X_val_scaled, y_val)
        val_loader = DataLoader(val_dataset, batch_size=256)

        # Get uncalibrated logits
        all_logits = []
        all_targets = []
        for model in ensemble.models:
            model.eval()
            model_logits = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    logits = model(inputs)
                    model_logits.append(logits.cpu())
                    if len(all_targets) < len(y_val):
                        all_targets.extend(targets.numpy())

            all_logits.append(torch.cat(model_logits))

        # Average logits
        avg_logits = torch.stack(all_logits).mean(0).numpy()
        all_targets = np.array(all_targets[: len(y_val)])

        # Fit temperature scaling
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(avg_logits, all_targets)

        # Compute calibration before and after
        uncal_probs = F.softmax(torch.FloatTensor(avg_logits), dim=1).numpy()
        cal_probs = F.softmax(temp_scaler(torch.FloatTensor(avg_logits)), dim=1).detach().numpy()

        before_cal = compute_calibration(all_targets, uncal_probs)
        after_cal = compute_calibration(all_targets, cal_probs)

        return {
            "temperature": float(temp_scaler.temperature.item()),
            "before_calibration": before_cal,
            "after_calibration": after_cal,
            "ece_improvement": float(before_cal["ece"] - after_cal["ece"]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    """Run the experiment with default configuration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    results = runner.run()

    print(f"\n{'=' * 60}")
    print("Experiment Results")
    print(f"{'=' * 60}")
    print(
        f"CV F1:        {results['cv_results']['mean_f1']:.4f} "
        f"± {results['cv_results']['std_f1']:.4f}"
    )
    print(f"Test F1:      {results['eval_results']['macro_f1']:.4f}")
    print(f"Ensemble Gain: +{results['eval_results']['ensemble_gain']:.4f}")
    print(f"ECE (before): {results['calibration']['before_calibration']['ece']:.4f}")
    print(f"ECE (after):  {results['calibration']['after_calibration']['ece']:.4f}")
    print(f"Temperature:  {results['calibration']['temperature']:.4f}")
    print(f"Total Time:   {results['total_time_seconds']:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
