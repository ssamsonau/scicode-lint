"""Experiment tracking with timestamps."""

from datetime import datetime


class ExperimentTracker:
    """Track experiments with timestamps."""

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.start_time = datetime.now()
        self.checkpoints = []

    def log_checkpoint(self, metrics: dict) -> None:
        """Log checkpoint with current timestamp."""
        self.checkpoints.append(
            {
                "time": datetime.now(),
                "metrics": metrics,
            }
        )

    def get_duration_seconds(self) -> float:
        """Get experiment duration."""
        return (datetime.now() - self.start_time).total_seconds()


def create_experiment_id() -> str:
    """Create experiment ID from timestamp."""
    return f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
