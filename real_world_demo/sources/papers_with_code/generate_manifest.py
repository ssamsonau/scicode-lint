"""Generate manifest.csv and organize collected files.

Copies qualifying files to collected_code/ and generates manifest with metadata.

Usage:
    python generate_manifest.py [--force]
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from real_world_demo.config import COLLECTED_DIR, DATA_DIR


def load_qualifying_files(input_file: Path) -> list[dict[str, Any]]:
    """Load qualifying files list from JSON.

    Args:
        input_file: Path to qualifying_files.json.

    Returns:
        List of file analysis dicts.

    Raises:
        FileNotFoundError: If input file doesn't exist.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Qualifying files not found: {input_file}. Run filter_files.py first."
        )

    with open(input_file) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def load_repos_metadata(papers_file: Path) -> dict[str, dict[str, Any]]:
    """Load repo metadata for enriching manifest from papers file.

    Args:
        papers_file: Path to ai_science_papers.json (has repo_urls embedded).

    Returns:
        Dict mapping repo_path to paper/repo metadata.
    """
    if not papers_file.exists():
        logger.warning(f"Papers file not found: {papers_file}")
        return {}

    with open(papers_file) as f:
        papers = json.load(f)

    # Build lookup by repo key (owner__repo)
    # repo_path in qualifying_files is like: cloned_repos/owner__repo
    metadata: dict[str, dict[str, Any]] = {}
    for paper in papers:
        for repo_url in paper.get("repo_urls", []):
            # Extract owner/repo from URL
            parts = repo_url.rstrip("/").split("/")
            if len(parts) >= 2:
                key = f"{parts[-2]}__{parts[-1]}"
                metadata[key] = {
                    "repo_url": repo_url,
                    "paper_url": paper.get("paper_url", ""),
                    "paper_title": paper.get("title", ""),
                    "domain": paper.get("matched_domain", ""),
                    "tasks": paper.get("tasks", []),
                    "arxiv_id": paper.get("arxiv_id", ""),
                }

    return metadata


def generate_unique_path(
    file_path: Path, repo_name: str, seen_per_repo: dict[str, set[str]]
) -> tuple[str, str]:
    """Generate a unique path for collected file within repo directory.

    Args:
        file_path: Original file path.
        repo_name: Name of the repo (owner__repo format).
        seen_per_repo: Dict mapping repo name to set of used filenames.

    Returns:
        Tuple of (repo_dir, filename) for the unique path.
    """
    # Initialize set for this repo if not exists
    if repo_name not in seen_per_repo:
        seen_per_repo[repo_name] = set()
    seen = seen_per_repo[repo_name]

    # Use original filename
    base_name = file_path.name

    # Ensure uniqueness within repo
    if base_name not in seen:
        seen.add(base_name)
        return repo_name, base_name

    # Add counter for duplicates
    counter = 1
    while True:
        stem = file_path.stem
        suffix = file_path.suffix
        unique_name = f"{stem}_{counter}{suffix}"
        if unique_name not in seen:
            seen.add(unique_name)
            return repo_name, unique_name
        counter += 1


def copy_files_and_generate_manifest(
    qualifying_files: list[dict[str, Any]],
    repos_metadata: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Copy files to output directory and generate manifest records.

    Files are organized by repo: files/owner__repo/filename.py

    Args:
        qualifying_files: List of file analysis dicts.
        repos_metadata: Dict mapping repo names to metadata.
        output_dir: Directory to copy files to.

    Returns:
        List of manifest records.
    """
    files_dir = output_dir / "files"
    files_dir.mkdir(exist_ok=True, parents=True)

    manifest_records = []
    seen_per_repo: dict[str, set[str]] = {}
    copied_count = 0
    skipped_count = 0

    for file_info in qualifying_files:
        original_path = Path(file_info["file_path"])
        repo_path = Path(file_info["repo_path"])
        repo_name = repo_path.name

        # Generate unique path within repo directory
        repo_dir, unique_filename = generate_unique_path(original_path, repo_name, seen_per_repo)
        repo_files_dir = files_dir / repo_dir
        repo_files_dir.mkdir(exist_ok=True, parents=True)
        dest_path = repo_files_dir / unique_filename

        # Copy file
        try:
            if not original_path.exists():
                logger.warning(f"File not found: {original_path}")
                skipped_count += 1
                continue

            shutil.copy2(original_path, dest_path)
            copied_count += 1
        except Exception as e:
            logger.warning(f"Error copying {original_path}: {e}")
            skipped_count += 1
            continue

        # Get repo metadata
        repo_meta = repos_metadata.get(repo_name, {})

        # Build manifest record
        record = {
            "file_path": str(dest_path.relative_to(output_dir)),
            "original_path": file_info["relative_path"],
            "repo_name": repo_name,
            "repo_url": repo_meta.get("repo_url", ""),
            "paper_url": repo_meta.get("paper_url", ""),
            "paper_title": repo_meta.get("paper_title", ""),
            "domain": repo_meta.get("domain", ""),
            "is_notebook": file_info.get("is_notebook", False),
            "file_size": file_info.get("file_size", 0),
            "line_count": file_info.get("line_count", 0),
            "ml_imports": ",".join(file_info.get("ml_imports", [])),
            "scientific_imports": ",".join(file_info.get("scientific_imports", [])),
        }
        manifest_records.append(record)

    logger.info(f"Copied {copied_count:,} files to {len(seen_per_repo):,} repo directories")
    logger.info(f"Skipped {skipped_count:,} files")
    return manifest_records


def save_manifest(records: list[dict[str, Any]], output_file: Path) -> None:
    """Save manifest to CSV file.

    Args:
        records: List of manifest records.
        output_file: Path to output CSV file.
    """
    if not records:
        logger.warning("No records to save!")
        return

    fieldnames = list(records[0].keys())

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Saved manifest with {len(records):,} entries to {output_file}")


def print_summary(records: list[dict[str, Any]]) -> None:
    """Print manifest summary statistics.

    Args:
        records: List of manifest records.
    """
    total = len(records)
    notebooks = sum(1 for r in records if r.get("is_notebook"))
    py_files = total - notebooks

    # Domain distribution
    domain_counts: dict[str, int] = {}
    for r in records:
        domain = r.get("domain", "unknown") or "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # ML library distribution
    ml_counts: dict[str, int] = {}
    for r in records:
        for lib in r.get("ml_imports", "").split(","):
            if lib:
                ml_counts[lib] = ml_counts.get(lib, 0) + 1

    # Unique repos
    repos = set(r.get("repo_name", "") for r in records)

    logger.info("=" * 50)
    logger.info("Manifest Summary:")
    logger.info(f"  Total files: {total:,}")
    logger.info(f"  Python files: {py_files:,}")
    logger.info(f"  Notebooks: {notebooks:,}")
    logger.info(f"  From {len(repos):,} repos")

    logger.info("  Domain distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {domain}: {count:,} ({100 * count / total:.1f}%)")

    logger.info("  ML library distribution:")
    for lib, count in sorted(ml_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"    {lib}: {count:,} ({100 * count / total:.1f}%)")


def main() -> None:
    """Main entry point for manifest generation."""
    parser = argparse.ArgumentParser(description="Generate manifest and collect files")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DATA_DIR / "pipeline_files.json",
        help="Input file with pipeline files list (from prefilter)",
    )
    parser.add_argument(
        "--papers-file",
        type=Path,
        default=DATA_DIR / "ai_science_papers.json",
        help="Papers file with embedded repo URLs (ai_science_papers.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COLLECTED_DIR,
        help="Output directory for collected files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if manifest exists",
    )
    args = parser.parse_args()

    manifest_file = args.output_dir / "manifest.csv"

    # Check if output already exists
    if manifest_file.exists() and not args.force:
        logger.info(f"Manifest already exists: {manifest_file}")
        logger.info("Use --force to regenerate, or delete the file manually")
        # Load and print summary
        import csv

        with open(manifest_file) as f:
            reader = csv.DictReader(f)
            existing_records = list(reader)
        logger.info(f"Existing manifest entries: {len(existing_records):,}")
        return

    # Load inputs
    qualifying_files = load_qualifying_files(args.input_file)
    logger.info(f"Loaded {len(qualifying_files):,} qualifying files")

    repos_metadata = load_repos_metadata(args.papers_file)
    logger.info(f"Loaded metadata for {len(repos_metadata):,} repos")

    # Generate manifest and copy files
    records = copy_files_and_generate_manifest(qualifying_files, repos_metadata, args.output_dir)

    # Save manifest
    save_manifest(records, manifest_file)

    # Print summary
    print_summary(records)


if __name__ == "__main__":
    main()
