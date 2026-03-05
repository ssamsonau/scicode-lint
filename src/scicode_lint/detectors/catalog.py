"""Load detection patterns from TOML pattern files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from scicode_lint.config import Severity


@dataclass
class DetectionPattern:
    """A single detection pattern from the catalog.

    Attributes:
        id: Pattern identifier (e.g., "ml-001", "pt-001")
        category: Pattern category (e.g., "ml-correctness", "pytorch")
        severity: Severity level (CRITICAL, HIGH, or MEDIUM)
        detection_question: What the pattern checks for (used in prompts)
        warning_message: Explanation of the issue and how to fix it
    """

    id: str
    category: str
    severity: Severity
    detection_question: str
    warning_message: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionPattern":
        """Create pattern from YAML dictionary.

        Args:
            data: Dictionary with pattern fields from YAML

        Returns:
            DetectionPattern instance
        """
        return cls(
            id=data["id"],
            category=data["category"],
            severity=Severity(data["severity"]),
            detection_question=data["detection_question"],
            warning_message=data["warning_message"],
        )


class DetectionCatalog:
    """Loads and manages detection patterns from TOML pattern files.

    Provides access to all detection patterns with methods to filter
    by ID, severity, or category.

    Example:
        >>> from scicode_lint import DetectionCatalog
        >>> from pathlib import Path
        >>> catalog = DetectionCatalog(Path("patterns"))
        >>> pattern = catalog.get_pattern("ml-001")
        >>> print(pattern.warning_message)
    """

    def __init__(self, patterns_dir: Optional[Path] = None):
        """Initialize catalog from TOML patterns directory.

        Args:
            patterns_dir: Path to patterns directory containing category subdirectories.
                         If None, uses default location.
        """
        if patterns_dir is None:
            # Use default patterns location
            from scicode_lint.config import get_default_patterns_dir

            patterns_dir = get_default_patterns_dir()

        self.patterns_dir = Path(patterns_dir)
        self.patterns = self._load_patterns()

    def _load_patterns(self) -> list[DetectionPattern]:
        """Load all patterns from TOML format."""
        from .pattern_loader import PatternLoader

        loader = PatternLoader(self.patterns_dir)
        toml_patterns = loader.load_all_patterns()

        # Convert to legacy DetectionPattern format
        return [loader.to_legacy_pattern(p) for p in toml_patterns]

    def get_pattern(self, pattern_id: str) -> DetectionPattern | None:
        """Get pattern by ID.

        Args:
            pattern_id: Pattern identifier (e.g., "ml-001", "pt-001")

        Returns:
            DetectionPattern object if found, None otherwise

        Example:
            >>> pattern = catalog.get_pattern("ml-001")
            >>> if pattern:
            ...     print(pattern.warning_message)
        """
        for pattern in self.patterns:
            if pattern.id == pattern_id:
                return pattern
        return None

    def get_patterns_by_severity(self, severity: Severity) -> list[DetectionPattern]:
        """Get all patterns matching a severity level.

        Args:
            severity: Severity level (Severity.CRITICAL, Severity.HIGH, or Severity.MEDIUM)

        Returns:
            List of DetectionPattern objects matching the severity

        Example:
            >>> from scicode_lint.config import Severity
            >>> critical = catalog.get_patterns_by_severity(Severity.CRITICAL)
            >>> print(f"Found {len(critical)} critical patterns")
        """
        return [p for p in self.patterns if p.severity == severity]

    def get_patterns_by_category(self, category: str) -> list[DetectionPattern]:
        """Get all patterns in a category.

        Args:
            category: Category name (e.g., "ml-correctness", "pytorch", "numerical")

        Returns:
            List of DetectionPattern objects in the category

        Example:
            >>> ml_patterns = catalog.get_patterns_by_category("ml-correctness")
            >>> for p in ml_patterns:
            ...     print(f"{p.id}: {p.warning_message}")
        """
        return [p for p in self.patterns if p.category == category]
