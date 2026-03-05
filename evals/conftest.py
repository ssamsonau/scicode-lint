"""Pytest configuration for scicode-lint evaluation tests."""

from pathlib import Path

import pytest
import yaml

from .run_eval import EvalRunner


@pytest.fixture(scope="session")
def patterns_dir():
    """Return path to patterns directory."""
    return Path(__file__).parent.parent / "patterns"


@pytest.fixture(scope="session")
def test_definitions_path():
    """Return path to test_definitions.yaml."""
    return Path(__file__).parent / "test_definitions.yaml"


@pytest.fixture(scope="session")
def eval_runner(patterns_dir, test_definitions_path):
    """Create EvalRunner instance for testing."""
    return EvalRunner(
        patterns_dir=patterns_dir,
        test_definitions_path=test_definitions_path,
        linter_timeout=300,
    )


@pytest.fixture(scope="session")
def test_config(test_definitions_path):
    """Load test configuration."""
    with open(test_definitions_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def enabled_patterns(test_config):
    """Get list of enabled pattern IDs."""
    return [p["id"] for p in test_config.get("patterns", []) if p.get("enabled", True)]


@pytest.fixture(scope="session")
def thresholds(test_config):
    """Get quality thresholds."""
    return test_config.get("thresholds", {})
