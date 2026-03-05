"""Configuration for scicode-lint with Pydantic validation."""

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# TOML loading (Python 3.11+ stdlib, fallback to tomli)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore[assignment]


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class LLMConfig(BaseSettings):
    """
    LLM client configuration with validation and env var support.

    vLLM server configuration (local or remote):
    - OPENAI_BASE_URL: vLLM server URL (e.g., http://localhost:5001/v1)
      Note: Uses OpenAI-compatible API format, but only works with vLLM servers

    Package-specific environment variables:
    - SCICODE_LINT_TEMPERATURE: Temperature 0.0-1.0 (default: 0.3)
    - SCICODE_LINT_TIMEOUT: Timeout in seconds (default: 120)
    - SCICODE_LINT_MAX_MODEL_LEN: Maximum context length (default: auto-detect)

    Note: All patterns are checked concurrently. vLLM handles batching and queuing internally.
    """

    base_url: str = Field(
        default="",
        description="API base URL (empty = auto-detect)",
    )
    model: str = Field(
        default="",
        description="Model name (empty = auto-detect)",
    )
    timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0.0=deterministic, 0.3=slight randomness)",
    )
    max_model_len: Optional[int] = Field(
        default=None,
        ge=1000,
        description="Maximum context length in tokens (None = auto-detect from server)",
    )

    model_config = SettingsConfigDict(
        env_prefix="SCICODE_LINT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def __init__(self, **data: Any) -> None:
        """Initialize and apply standard OpenAI env vars if present."""
        import os

        # Standard OpenAI env vars take precedence
        if not data.get("base_url") and os.getenv("OPENAI_BASE_URL"):
            data["base_url"] = os.getenv("OPENAI_BASE_URL")
        super().__init__(**data)


class LinterConfig:
    """Linter configuration (not using Pydantic to keep simple)."""

    def __init__(
        self,
        patterns_dir: Optional[Path] = None,
        llm_config: Optional[LLMConfig] = None,
        min_confidence: float = 0.7,
        enabled_severities: Optional[set[Severity]] = None,
        enabled_patterns: Optional[set[str]] = None,
        enabled_categories: Optional[set[str]] = None,
    ):
        self.patterns_dir = patterns_dir or get_default_patterns_dir()
        self.llm_config = llm_config or load_llm_config()
        self.min_confidence = min_confidence
        self.enabled_severities = enabled_severities or {
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
        }
        self.enabled_patterns = enabled_patterns
        self.enabled_categories = enabled_categories


def get_default_patterns_dir() -> Path:
    """
    Get path to patterns directory.

    Searches in order:
    1. patterns/ (from package root)
    2. ~/.config/scicode-lint/patterns (user patterns)

    Returns:
        Path to patterns directory
    """
    # Priority 1: Package patterns
    pkg_patterns = Path(__file__).parent.parent.parent / "patterns"
    if pkg_patterns.exists():
        return pkg_patterns

    # Priority 2: User patterns directory
    user_patterns = Path.home() / ".config" / "scicode-lint" / "patterns"
    if user_patterns.exists():
        return user_patterns

    # Fallback to package location even if it doesn't exist yet
    return pkg_patterns


def load_config_from_toml() -> dict[str, Any]:
    """
    Load configuration from config.toml.

    Searches in order:
    1. $SCICODE_LINT_CONFIG (env var pointing to config file)
    2. ./config.toml (current directory)
    3. ~/.config/scicode-lint/config.toml (user config)
    4. Package default (built-in)

    Returns:
        Parsed config dict (empty if no TOML support or file not found)
    """
    if not tomllib:
        return {}

    import os

    # Priority 1: Environment variable
    config_path_env = os.getenv("SCICODE_LINT_CONFIG")
    if config_path_env:
        config_path = Path(config_path_env)
        if config_path.exists():
            with open(config_path, "rb") as f:
                return tomllib.load(f)

    # Priority 2: Current directory
    config_path = Path("config.toml")
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    # Priority 3: User config directory
    user_config = Path.home() / ".config" / "scicode-lint" / "config.toml"
    if user_config.exists():
        with open(user_config, "rb") as f:
            return tomllib.load(f)

    # Priority 4: Package default
    package_config = Path(__file__).parent.parent.parent / "config.toml"
    if package_config.exists():
        with open(package_config, "rb") as f:
            return tomllib.load(f)

    return {}


def load_llm_config() -> LLMConfig:
    """
    Load LLM configuration from TOML file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables (SCICODE_LINT_*)
    2. .env file
    3. config.toml file
    4. Defaults

    Returns:
        Validated LLMConfig instance
    """
    # Load TOML config
    toml_config = load_config_from_toml()
    llm_toml = toml_config.get("llm", {})

    # Pydantic Settings will automatically:
    # - Load from TOML values (passed as kwargs)
    # - Override with env vars (SCICODE_LINT_*)
    # - Validate types and constraints
    return LLMConfig(**llm_toml)


def get_default_config() -> LinterConfig:
    """Get default linter configuration with TOML/env overrides."""
    return LinterConfig(
        patterns_dir=get_default_patterns_dir(),
        llm_config=load_llm_config(),
    )
