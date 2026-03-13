"""PapersWithCode data source: scientific ML repositories from PWC archive.

Downloads papers and code repositories from PapersWithCode via HuggingFace datasets,
filters by scientific domains, clones repositories, and prepares files for analysis.

Data sources:
- Papers: https://huggingface.co/datasets/pwc-archive/papers-with-abstracts
- Links: https://huggingface.co/datasets/pwc-archive/links-between-paper-and-code

Usage:
    # Full pipeline (download + clone + prepare)
    python -m real_world_demo.sources.papers_with_code --run --papers 50

    # Step by step:
    python -m real_world_demo.sources.papers_with_code --download --papers 50
    python -m real_world_demo.sources.papers_with_code --clone
    python -m real_world_demo.sources.papers_with_code --prepare

    # Then analyze with:
    python -m real_world_demo.run_analysis \
        --manifest real_world_demo/collected_code/manifest.csv \
        --base-dir real_world_demo/collected_code
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

from real_world_demo.config import CLONED_DIR, COLLECTED_DIR, DATA_DIR


def download_papers(
    limit: int | None = None,
    domains: list[str] | None = None,
    force: bool = False,
    balanced: bool = True,
) -> Path:
    """Download and filter papers from PapersWithCode via HuggingFace.

    Args:
        limit: Maximum number of papers to include.
        domains: List of scientific domains to filter (None = all).
        force: If True, re-download even if data exists.
        balanced: If True, sample evenly across domains.

    Returns:
        Path to filtered_papers.json file.
    """
    papers_file = DATA_DIR / "filtered_papers.json"

    if papers_file.exists() and not force:
        with open(papers_file) as f:
            papers = json.load(f)
        logger.info(f"Papers already downloaded: {papers_file} ({len(papers)} papers)")
        return papers_file

    # Build command
    cmd = [sys.executable, "-m", "real_world_demo.sources.papers_with_code.filter_papers"]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if domains:
        cmd.extend(["--domains", ",".join(domains)])
    if force:
        cmd.append("--force")
    if not balanced:
        cmd.append("--no-balanced")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"filter_papers failed with exit code {result.returncode}")

    return papers_file


def clone_repos(
    max_repos: int | None = None,
    max_concurrent: int = 3,
    force: bool = False,
) -> Path:
    """Clone repositories from ai_science_papers.json.

    Args:
        max_repos: Maximum number of repos to clone.
        max_concurrent: Maximum concurrent clone operations.
        force: If True, re-clone even if repos exist.

    Returns:
        Path to clone_results.json file.
    """
    results_file = DATA_DIR / "clone_results.json"
    papers_file = DATA_DIR / "ai_science_papers.json"

    if not papers_file.exists():
        raise FileNotFoundError(
            f"AI+science papers file not found: {papers_file}. "
            "Run filter_papers and filter_abstracts first (or use run_pipeline.py)."
        )

    if results_file.exists() and not force:
        with open(results_file) as f:
            results = json.load(f)
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"Repos already cloned: {results_file} ({success_count} successful)")
        return results_file

    # Build command
    cmd = [sys.executable, "-m", "real_world_demo.sources.papers_with_code.clone_repos"]
    if max_repos:
        cmd.extend(["--limit", str(max_repos)])
    cmd.extend(["--max-concurrent", str(max_concurrent)])
    if force:
        cmd.append("--force")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"clone_repos failed with exit code {result.returncode}")

    return results_file


def filter_files(force: bool = False) -> Path:
    """Filter Python files from cloned repositories.

    Args:
        force: If True, re-filter even if output exists.

    Returns:
        Path to qualifying_files.json file.
    """
    output_file = DATA_DIR / "qualifying_files.json"

    if not CLONED_DIR.exists() or not list(CLONED_DIR.iterdir()):
        raise FileNotFoundError(f"No cloned repos found: {CLONED_DIR}. Run --clone first.")

    if output_file.exists() and not force:
        with open(output_file) as f:
            files = json.load(f)
        logger.info(f"Files already filtered: {output_file} ({len(files)} files)")
        return output_file

    # Build command
    cmd = [sys.executable, "-m", "real_world_demo.sources.papers_with_code.filter_files"]
    if force:
        cmd.append("--force")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"filter_files failed with exit code {result.returncode}")

    return output_file


def prepare_manifest(force: bool = False) -> Path:
    """Generate manifest.csv from filtered files.

    Args:
        force: If True, regenerate even if manifest exists.

    Returns:
        Path to manifest.csv file.
    """
    manifest_file = COLLECTED_DIR / "manifest.csv"
    qualifying_file = DATA_DIR / "qualifying_files.json"

    if not qualifying_file.exists():
        raise FileNotFoundError(
            f"Qualifying files not found: {qualifying_file}. "
            "Run filter_files first (or use --clone --prepare)."
        )

    if manifest_file.exists() and not force:
        # Count lines (excluding header)
        with open(manifest_file) as f:
            line_count = sum(1 for _ in f) - 1
        logger.info(f"Manifest already exists: {manifest_file} ({line_count} files)")
        return manifest_file

    # Build command
    cmd = [sys.executable, "-m", "real_world_demo.sources.papers_with_code.generate_manifest"]
    if force:
        cmd.append("--force")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"generate_manifest failed with exit code {result.returncode}")

    return manifest_file


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download and prepare PapersWithCode repositories")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and filter papers from HuggingFace",
    )
    parser.add_argument(
        "--clone",
        action="store_true",
        help="Clone repositories",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Filter files and generate manifest",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run full pipeline (download + clone + prepare)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/re-clone/re-prepare",
    )
    parser.add_argument(
        "--papers",
        type=int,
        help="Limit number of papers to download",
    )
    parser.add_argument(
        "--domains",
        type=str,
        help="Comma-separated list of scientific domains to filter",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent clone operations (default: 3)",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="Disable balanced sampling across domains",
    )
    args = parser.parse_args()

    # Default to showing help if no action specified
    if not (args.download or args.clone or args.prepare or args.run):
        parser.print_help()
        return

    domains = args.domains.split(",") if args.domains else None

    if args.download or args.run:
        download_papers(
            limit=args.papers,
            domains=domains,
            force=args.force,
            balanced=not args.no_balanced,
        )

    if args.clone or args.run:
        clone_repos(
            max_repos=args.papers,
            max_concurrent=args.max_concurrent,
            force=args.force,
        )

    if args.prepare or args.run:
        filter_files(force=args.force)
        manifest_path = prepare_manifest(force=args.force)

        logger.info("=" * 50)
        logger.info("To run analysis:")
        logger.info("  python -m real_world_demo.run_analysis \\")
        logger.info(f"    --manifest {manifest_path} \\")
        logger.info(f"    --base-dir {COLLECTED_DIR}")


if __name__ == "__main__":
    main()
