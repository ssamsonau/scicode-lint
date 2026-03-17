"""Datetime for directory naming and console logging."""

import os
from datetime import datetime


def save_results(data, base_dir="results"):
    """Save results to timestamped directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(str(data))

    return output_dir


def log_training_progress(epoch, loss):
    """Log training progress with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Epoch {epoch}: loss = {loss:.4f}")
