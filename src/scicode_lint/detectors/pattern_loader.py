"""
Pattern loader for TOML-based patterns.

Loads and validates pattern definitions from pattern.toml files.
"""

import tomllib
from pathlib import Path

from loguru import logger

from scicode_lint.config import Severity

from .catalog import DetectionPattern
from .pattern_models import PatternTOML


class PatternLoader:
    """Load patterns from TOML files."""

    def __init__(self, patterns_dir: Path):
        """
        Initialize pattern loader.

        Args:
            patterns_dir: Root directory containing category subdirectories
        """
        self.patterns_dir = Path(patterns_dir)

    def load_pattern_toml(self, pattern_path: Path) -> PatternTOML:
        """
        Load and validate a pattern.toml file.

        Args:
            pattern_path: Directory containing pattern.toml

        Returns:
            Validated PatternTOML object

        Raises:
            FileNotFoundError: If pattern.toml doesn't exist
            ValidationError: If TOML structure is invalid
        """
        toml_file = pattern_path / "pattern.toml"
        if not toml_file.exists():
            raise FileNotFoundError(f"No pattern.toml found in {pattern_path}")

        with open(toml_file, "rb") as f:
            data = tomllib.load(f)

        return PatternTOML.model_validate(data)

    def load_all_patterns(self) -> list[PatternTOML]:
        """
        Load all patterns from category directories.

        Scans patterns_dir for category directories, then loads all
        patterns within each category.

        Returns:
            List of validated PatternTOML objects

        Raises:
            ValidationError: If any pattern is invalid
        """
        patterns: list[PatternTOML] = []

        if not self.patterns_dir.exists():
            logger.warning(f"Patterns directory not found: {self.patterns_dir}")
            return patterns

        for category_dir in self.patterns_dir.iterdir():
            # Skip non-directories and special directories
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue

            for pattern_dir in category_dir.iterdir():
                if not pattern_dir.is_dir():
                    continue

                pattern_file = pattern_dir / "pattern.toml"
                if pattern_file.exists():
                    try:
                        patterns.append(self.load_pattern_toml(pattern_dir))
                    except Exception as e:
                        # Log but continue loading other patterns
                        logger.warning(f"Failed to load {pattern_dir}: {e}")
                        continue

        return patterns

    def to_detection_pattern(self, pattern_toml: PatternTOML) -> DetectionPattern:
        """
        Convert TOML pattern to DetectionPattern dataclass.

        Args:
            pattern_toml: Validated PatternTOML object

        Returns:
            DetectionPattern object
        """
        return DetectionPattern(
            id=pattern_toml.meta.id,
            category=pattern_toml.meta.category,
            severity=Severity(pattern_toml.meta.severity),
            detection_question=pattern_toml.detection.question,
            warning_message=pattern_toml.detection.warning_message,
        )

    def find_pattern_by_id(self, pattern_id: str) -> PatternTOML:
        """
        Find a specific pattern by ID.

        Args:
            pattern_id: Pattern identifier (e.g., "ml-001")

        Returns:
            PatternTOML object

        Raises:
            ValueError: If pattern not found
        """
        all_patterns = self.load_all_patterns()

        for pattern in all_patterns:
            if pattern.meta.id == pattern_id:
                return pattern

        # Build helpful error with available patterns
        available_ids = sorted([p.meta.id for p in all_patterns])
        available_str = ", ".join(available_ids[:10])
        if len(available_ids) > 10:
            available_str += f", ... ({len(available_ids)} total)"

        raise ValueError(
            f"Pattern not found: {pattern_id}\n\n"
            f"Available patterns: {available_str}\n"
            f"Suggestions:\n"
            f"  • Check pattern ID spelling\n"
            f"  • List all patterns: linter.list_patterns()\n"
            f"  • Check patterns directory: {self.patterns_dir}"
        )
