"""Shared config loading for dev tooling.

Loads config.toml from src/scicode_lint/ — the single source of truth.
Used by pattern_verification/, evals/, and real_world_demo/.
"""

import tomllib
from pathlib import Path
from typing import Any


def _find_config_path() -> Path:
    """Find config.toml by walking up from this file.

    Returns:
        Path to config.toml

    Raises:
        FileNotFoundError: If config.toml cannot be found.
    """
    # dev_lib/ is at repo root, config is at src/scicode_lint/config.toml
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "src" / "scicode_lint" / "config.toml"
    if config_path.exists():
        return config_path

    raise FileNotFoundError(
        f"Config file not found: {config_path}\nRun from the repository root directory."
    )


def load_project_config() -> dict[str, Any]:
    """Load config.toml from src/scicode_lint/.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If config.toml cannot be found.
        RuntimeError: If required sections are missing.
    """
    config_path = _find_config_path()
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if "pattern_verification" not in config:
        raise RuntimeError(f"Config file missing [pattern_verification] section: {config_path}")
    return config
