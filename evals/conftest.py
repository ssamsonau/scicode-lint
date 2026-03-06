"""Pytest configuration for scicode-lint evaluation tests."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from .run_eval import EvalRunner


@pytest.fixture(scope="session")
def patterns_dir() -> Path:
    """Return path to patterns directory."""
    return Path(__file__).parent.parent / "patterns"


@pytest.fixture(scope="session")
def test_definitions_path() -> Path:
    """Return path to test_definitions.yaml."""
    return Path(__file__).parent / "test_definitions.yaml"


@pytest.fixture(scope="session")
def eval_runner(patterns_dir: Path, test_definitions_path: Path) -> EvalRunner:
    """Create EvalRunner instance for testing."""
    return EvalRunner(
        patterns_dir=patterns_dir,
        test_definitions_path=test_definitions_path,
        linter_timeout=300,
    )


@pytest.fixture(scope="session")
def test_config(test_definitions_path: Path) -> dict[str, Any]:
    """Load test configuration."""
    with open(test_definitions_path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def enabled_patterns(test_config: dict[str, Any]) -> list[str]:
    """Get list of enabled pattern IDs."""
    return [p["id"] for p in test_config.get("patterns", []) if p.get("enabled", True)]


@pytest.fixture(scope="session")
def thresholds(test_config: dict[str, Any]) -> dict[str, Any]:
    """Get quality thresholds."""
    result: dict[str, Any] = test_config.get("thresholds", {})
    return result
