"""Filter Python files containing ML code from cloned repositories.

Scans cloned repos for .py and .ipynb files with ML library imports.

Usage:
    python filter_files.py [--min-size 500]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from real_world_demo.config import (
    CLONED_DIR,
    DATA_DIR,
    EXCLUDE_FILES,
    EXCLUDE_PATTERNS,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE_BYTES,
    MIN_LINES,
    ML_IMPORTS,
    SCIENTIFIC_IMPORTS,
)


def matches_exclude_pattern(file_path: Path, repo_root: Path) -> bool:
    """Check if file matches any exclusion pattern.

    Args:
        file_path: Absolute path to file.
        repo_root: Root directory of the repo.

    Returns:
        True if file should be excluded.
    """
    # Check filename against exclude list
    if file_path.name in EXCLUDE_FILES:
        return True

    # Get path relative to repo root for pattern matching
    try:
        rel_path = str(file_path.relative_to(repo_root))
    except ValueError:
        rel_path = file_path.name

    # Check against exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        # Convert glob pattern to regex
        regex = pattern.replace("*", ".*").replace("/", r"[/\\]")
        if re.match(regex, rel_path):
            return True

    return False


def extract_imports(content: str) -> set[str]:
    """Extract import statements from Python code.

    Args:
        content: Python source code.

    Returns:
        Set of imported module names.
    """
    imports: set[str] = set()

    # Match import statements (simple regex, not full parse)
    # import X, from X import Y
    import_patterns = [
        r"^\s*import\s+(\w+)",  # import foo
        r"^\s*from\s+(\w+)",  # from foo import bar
    ]

    for pattern in import_patterns:
        for match in re.finditer(pattern, content, re.MULTILINE):
            imports.add(match.group(1))

    return imports


def has_ml_imports(imports: set[str]) -> tuple[bool, list[str]]:
    """Check if imports include ML libraries.

    Args:
        imports: Set of imported module names.

    Returns:
        Tuple of (has_ml_imports, list of matched ML imports).
    """
    matched = []
    for lib in ML_IMPORTS:
        # Handle aliases like scikit-learn -> sklearn
        if lib == "scikit-learn" and "sklearn" in imports:
            matched.append("sklearn")
        elif lib in imports:
            matched.append(lib)

    return len(matched) > 0, matched


def has_scientific_imports(imports: set[str]) -> tuple[bool, list[str]]:
    """Check if imports include scientific computing libraries.

    Args:
        imports: Set of imported module names.

    Returns:
        Tuple of (has_scientific_imports, list of matched imports).
    """
    matched = [lib for lib in SCIENTIFIC_IMPORTS if lib in imports]
    return len(matched) > 0, matched


def read_python_file(file_path: Path) -> str | None:
    """Read Python file content.

    Args:
        file_path: Path to .py file.

    Returns:
        File content or None if unreadable.
    """
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.debug(f"Error reading {file_path}: {e}")
        return None


def read_notebook_code(file_path: Path) -> str | None:
    """Extract code cells from Jupyter notebook.

    Args:
        file_path: Path to .ipynb file.

    Returns:
        Concatenated code from code cells or None if unreadable.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            notebook = json.load(f)

        code_cells = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                if isinstance(source, list):
                    code_cells.append("".join(source))
                else:
                    code_cells.append(source)

        return "\n\n".join(code_cells)
    except Exception as e:
        logger.debug(f"Error reading notebook {file_path}: {e}")
        return None


def analyze_file(
    file_path: Path, repo_root: Path, min_size: int, max_size: int, min_lines: int
) -> dict[str, Any] | None:
    """Analyze a single Python file for ML content.

    Args:
        file_path: Path to file.
        repo_root: Root directory of the repo.
        min_size: Minimum file size in bytes.
        max_size: Maximum file size in bytes.
        min_lines: Minimum number of lines.

    Returns:
        File analysis dict if qualifies, None otherwise.
    """
    # Check file size
    try:
        file_size = file_path.stat().st_size
    except OSError:
        return None

    if file_size < min_size or file_size > max_size:
        return None

    # Check exclusion patterns
    if matches_exclude_pattern(file_path, repo_root):
        return None

    # Read file content
    is_notebook = file_path.suffix == ".ipynb"
    if is_notebook:
        content = read_notebook_code(file_path)
    else:
        content = read_python_file(file_path)

    if not content:
        return None

    # Check line count
    lines = content.count("\n") + 1
    if lines < min_lines:
        return None

    # Extract and check imports
    imports = extract_imports(content)
    has_ml, ml_libs = has_ml_imports(imports)
    has_sci, sci_libs = has_scientific_imports(imports)

    # Must have at least one ML import
    if not has_ml:
        return None

    return {
        "file_path": str(file_path),
        "relative_path": str(file_path.relative_to(repo_root)),
        "repo_path": str(repo_root),
        "is_notebook": is_notebook,
        "file_size": file_size,
        "line_count": lines,
        "ml_imports": ml_libs,
        "scientific_imports": sci_libs,
    }


def scan_repo(
    repo_path: Path, min_size: int, max_size: int, min_lines: int
) -> list[dict[str, Any]]:
    """Scan a repository for qualifying Python files.

    Args:
        repo_path: Path to cloned repository.
        min_size: Minimum file size in bytes.
        max_size: Maximum file size in bytes.
        min_lines: Minimum number of lines.

    Returns:
        List of qualifying file analysis dicts.
    """
    qualifying_files = []

    # Find all Python files
    for pattern in ["**/*.py", "**/*.ipynb"]:
        for file_path in repo_path.glob(pattern):
            if file_path.is_file():
                result = analyze_file(file_path, repo_path, min_size, max_size, min_lines)
                if result:
                    qualifying_files.append(result)

    return qualifying_files


def filter_all_repos(
    cloned_dir: Path,
    min_size: int = MIN_FILE_SIZE_BYTES,
    max_size: int = MAX_FILE_SIZE_BYTES,
    min_lines: int = MIN_LINES,
) -> list[dict[str, Any]]:
    """Filter files from all cloned repositories.

    Args:
        cloned_dir: Directory containing cloned repos.
        min_size: Minimum file size in bytes.
        max_size: Maximum file size in bytes.
        min_lines: Minimum number of lines.

    Returns:
        List of qualifying file analysis dicts across all repos.
    """
    all_files = []

    # Iterate over all cloned repos
    repo_dirs = [d for d in cloned_dir.iterdir() if d.is_dir() and (d / ".git").exists()]
    logger.info(f"Scanning {len(repo_dirs):,} cloned repos...")

    for i, repo_path in enumerate(repo_dirs):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(repo_dirs)} repos scanned")

        files = scan_repo(repo_path, min_size, max_size, min_lines)
        if files:
            logger.debug(f"{repo_path.name}: {len(files)} qualifying files")
            all_files.extend(files)

    return all_files


def save_results(qualifying_files: list[dict[str, Any]], output_file: Path) -> None:
    """Save qualifying files list to JSON.

    Args:
        qualifying_files: List of file analysis dicts.
        output_file: Path to output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(qualifying_files, f, indent=2)
    logger.info(f"Saved {len(qualifying_files):,} qualifying files to {output_file}")


def print_summary(qualifying_files: list[dict[str, Any]]) -> None:
    """Print summary statistics.

    Args:
        qualifying_files: List of file analysis dicts.
    """
    total = len(qualifying_files)
    notebooks = sum(1 for f in qualifying_files if f.get("is_notebook"))
    py_files = total - notebooks

    # Count by ML library
    ml_counts: dict[str, int] = {}
    for f in qualifying_files:
        for lib in f.get("ml_imports", []):
            ml_counts[lib] = ml_counts.get(lib, 0) + 1

    # Count repos
    repos = set(f.get("repo_path", "") for f in qualifying_files)

    logger.info("=" * 50)
    logger.info("Filter Summary:")
    logger.info(f"  Total qualifying files: {total:,}")
    logger.info(f"  Python files: {py_files:,}")
    logger.info(f"  Notebooks: {notebooks:,}")
    logger.info(f"  From {len(repos):,} repos")
    logger.info("  ML library distribution:")
    for lib, count in sorted(ml_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {lib}: {count:,} ({100 * count / total:.1f}%)")


def main() -> None:
    """Main entry point for file filtering."""
    parser = argparse.ArgumentParser(description="Filter ML Python files from cloned repos")
    parser.add_argument(
        "--min-size",
        type=int,
        default=MIN_FILE_SIZE_BYTES,
        help=f"Minimum file size in bytes (default: {MIN_FILE_SIZE_BYTES})",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=MAX_FILE_SIZE_BYTES,
        help=f"Maximum file size in bytes (default: {MAX_FILE_SIZE_BYTES})",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=MIN_LINES,
        help=f"Minimum number of lines (default: {MIN_LINES})",
    )
    parser.add_argument(
        "--cloned-dir",
        type=Path,
        default=CLONED_DIR,
        help="Directory containing cloned repos",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DATA_DIR / "qualifying_files.json",
        help="Output file for qualifying files list",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-filtering even if output file exists",
    )
    args = parser.parse_args()

    # Check if output already exists
    if args.output_file.exists() and not args.force:
        logger.info(f"Output already exists: {args.output_file}")
        logger.info("Use --force to re-filter, or delete the file manually")
        # Load and print summary of existing data
        with open(args.output_file) as f:
            existing_files = json.load(f)
        logger.info(f"Existing qualifying files: {len(existing_files):,}")
        return

    if not args.cloned_dir.exists():
        logger.error(f"Cloned repos directory not found: {args.cloned_dir}")
        logger.error("Run clone_repos.py first.")
        return

    # Filter files
    qualifying_files = filter_all_repos(
        args.cloned_dir,
        min_size=args.min_size,
        max_size=args.max_size,
        min_lines=args.min_lines,
    )

    if not qualifying_files:
        logger.warning("No qualifying files found!")
        return

    # Save results
    save_results(qualifying_files, args.output_file)

    # Print summary
    print_summary(qualifying_files)


if __name__ == "__main__":
    main()
