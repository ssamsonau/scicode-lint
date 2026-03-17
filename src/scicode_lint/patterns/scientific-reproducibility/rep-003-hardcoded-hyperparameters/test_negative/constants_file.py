"""Configuration-driven experiment using YAML config files."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    """Configuration loaded from external YAML file."""

    data_path: Path
    output_path: Path
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 64])
    dropout_rate: float = 0.5

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ExperimentConfig":
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config_dict["data_path"] = Path(config_dict["data_path"])
        config_dict["output_path"] = Path(config_dict["output_path"])
        return cls(**config_dict)


def run_experiment(config: ExperimentConfig):
    """Run experiment using external configuration."""
    print(f"Loading data from {config.data_path}")
    print(f"Training with lr={config.learning_rate}, batch_size={config.batch_size}")
    print(f"Model architecture: {config.hidden_dims}")
    print(f"Results will be saved to {config.output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(Path(args.config))
    run_experiment(config)
