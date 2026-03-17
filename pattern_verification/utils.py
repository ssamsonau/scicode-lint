"""Shared utilities for pattern verification tools."""

from pathlib import Path


def find_all_patterns(patterns_dir: Path) -> list[Path]:
    """Find all pattern directories containing a pattern.toml file.

    Args:
        patterns_dir: Root directory containing pattern categories

    Returns:
        List of pattern directory paths, sorted alphabetically
    """
    patterns = []
    for category in patterns_dir.iterdir():
        if not category.is_dir() or category.name.startswith((".", "_")):
            continue
        for pattern_dir in category.iterdir():
            if not pattern_dir.is_dir() or pattern_dir.name.startswith((".", "_")):
                continue
            if (pattern_dir / "pattern.toml").exists():
                patterns.append(pattern_dir)
    return sorted(patterns)


def resolve_pattern(patterns_dir: Path, pattern_id: str) -> list[Path]:
    """Resolve a pattern ID to matching pattern directories.

    Supports both exact directory names (e.g., "ml-001-scaler-leakage")
    and short ID prefixes (e.g., "ml-001").

    Args:
        patterns_dir: Root directory containing pattern categories
        pattern_id: Full directory name or short ID prefix

    Returns:
        List of matching pattern directory paths (usually 1).
    """
    # Try exact match first (fastest path)
    exact = list(patterns_dir.glob(f"*/{pattern_id}"))
    if exact:
        return exact

    # Try prefix match: "ml-001" matches "ml-001-scaler-leakage"
    matches = []
    for pattern_dir in find_all_patterns(patterns_dir):
        if pattern_dir.name.startswith(pattern_id + "-") or pattern_dir.name == pattern_id:
            matches.append(pattern_dir)
    return matches
