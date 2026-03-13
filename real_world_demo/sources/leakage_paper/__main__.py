"""Leakage paper data source: sample notebooks from Yang et al. ASE'22.

Downloads and prepares the sample notebooks dataset from the paper
"Data Leakage in Notebooks: Static Detection and Better Processes".

Paper: https://arxiv.org/abs/2209.03345
Repository: https://github.com/malusamayo/GitHubAPI-Crawler

Usage:
    # Full pipeline (download + prepare)
    python -m real_world_demo.sources.leakage_paper --run

    # Step by step:
    python -m real_world_demo.sources.leakage_paper --download
    python -m real_world_demo.sources.leakage_paper --prepare

    # Then analyze with:
    python -m real_world_demo.run_analysis \
        --manifest real_world_demo/data/leakage_paper/manifest.csv \
        --base-dir real_world_demo/collected_code/leakage_paper
"""

import argparse
import csv
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from loguru import logger

from real_world_demo.config import (
    LEAKAGE_PAPER_COLLECTED_DIR,
    LEAKAGE_PAPER_DATA_DIR,
    ML_IMPORTS,
    SCIENTIFIC_IMPORTS,
)
from real_world_demo.utils import NotebookParseError, extract_code_from_notebook

# URLs for data download
GITHUB_RAW_BASE = "https://github.com/malusamayo/GitHubAPI-Crawler/raw/master/evaluation_materials"
SAMPLE_NOTEBOOKS_URL = f"{GITHUB_RAW_BASE}/sample-notebooks.zip"
GROUND_TRUTH_URL = f"{GITHUB_RAW_BASE}/manual-labels.csv"

# Paper metadata
PAPER_URL = "https://arxiv.org/abs/2209.03345"
PAPER_TITLE = "Data Leakage in Notebooks: Static Detection and Better Processes"
REPO_URL = "https://github.com/malusamayo/GitHubAPI-Crawler"
REPO_NAME = "leakage_paper_sample"


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a file from URL to destination.

    Args:
        url: URL to download from.
        dest: Destination file path.
        force: If True, re-download even if file exists.

    Returns:
        Path to downloaded file.
    """
    if dest.exists() and not force:
        logger.info(f"File already exists: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url}...")

    urllib.request.urlretrieve(url, dest)
    logger.info(f"Downloaded to {dest}")
    return dest


def download_data(force: bool = False) -> Path:
    """Download and extract sample-notebooks.zip.

    Args:
        force: If True, re-download and re-extract.

    Returns:
        Path to extracted raw directory.
    """
    raw_dir = LEAKAGE_PAPER_DATA_DIR / "raw"
    zip_path = LEAKAGE_PAPER_DATA_DIR / "sample-notebooks.zip"

    # Check if already extracted (look for .ipynb files)
    if raw_dir.exists() and list(raw_dir.glob("**/*.ipynb")) and not force:
        ipynb_count = len(list(raw_dir.glob("**/*.ipynb")))
        logger.info(f"Data already extracted: {raw_dir} ({ipynb_count} .ipynb files)")
        return raw_dir

    # Download zip
    download_file(SAMPLE_NOTEBOOKS_URL, zip_path, force=force)

    # Extract
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting to {raw_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    ipynb_count = len(list(raw_dir.glob("**/*.ipynb")))
    logger.info(f"Extracted {ipynb_count} .ipynb files")
    return raw_dir


def download_ground_truth(force: bool = False) -> Path:
    """Download manual-labels.csv ground truth file.

    Args:
        force: If True, re-download even if file exists.

    Returns:
        Path to ground truth CSV file.
    """
    dest = LEAKAGE_PAPER_DATA_DIR / "ground_truth.csv"
    return download_file(GROUND_TRUTH_URL, dest, force=force)


def detect_imports(content: str) -> tuple[list[str], list[str]]:
    """Extract ML and scientific imports from file content.

    Args:
        content: Python file content.

    Returns:
        Tuple of (ml_imports, scientific_imports) found in the file.
    """
    ml_found = []
    scientific_found = []

    # Simple regex for import statements
    import_pattern = r"(?:from\s+(\w+)|import\s+(\w+))"

    for match in re.finditer(import_pattern, content):
        module = match.group(1) or match.group(2)
        if module:
            module_lower = module.lower()
            if module_lower in [imp.lower() for imp in ML_IMPORTS]:
                ml_found.append(module)
            if module_lower in [imp.lower() for imp in SCIENTIFIC_IMPORTS]:
                scientific_found.append(module)

    return list(set(ml_found)), list(set(scientific_found))


def prepare_manifest(max_files: int | None = None) -> Path:
    """Generate manifest.csv from extracted notebooks.

    Copies .ipynb files to collected_code directory and creates manifest.
    scicode-lint can analyze .ipynb files directly.

    Args:
        max_files: Optional limit on number of files.

    Returns:
        Path to generated manifest.csv.
    """
    raw_dir = LEAKAGE_PAPER_DATA_DIR / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_dir}. Run --download first.")

    # Destination for collected files
    files_dir = LEAKAGE_PAPER_COLLECTED_DIR / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    # Find all Jupyter notebook files
    notebook_files = sorted(raw_dir.glob("**/*.ipynb"))
    logger.info(f"Found {len(notebook_files)} .ipynb files")

    if max_files:
        notebook_files = notebook_files[:max_files]
        logger.info(f"Limited to {max_files} files")

    # Generate manifest
    manifest_path = LEAKAGE_PAPER_DATA_DIR / "manifest.csv"
    records = []

    for nb_file in notebook_files:
        # Extract code for import detection and line counting
        try:
            content = extract_code_from_notebook(nb_file)
        except NotebookParseError as e:
            logger.warning(f"Skipping unparseable notebook: {e}")
            continue
        if not content.strip():
            logger.debug(f"Skipping empty notebook: {nb_file.name}")
            continue

        # Generate output filename (extract nb_XXX from filename like 2021-09-11-nb_605.ipynb)
        match = re.search(r"nb_(\d+)", nb_file.stem)
        if match:
            nb_id = match.group(1)
            dest_name = f"nb_{nb_id}.ipynb"
        else:
            dest_name = f"{nb_file.stem}.ipynb"

        dest_path = files_dir / dest_name

        # Handle duplicate names
        counter = 1
        while dest_path.exists():
            if match:
                dest_name = f"nb_{nb_id}_{counter}.ipynb"
            else:
                dest_name = f"{nb_file.stem}_{counter}.ipynb"
            dest_path = files_dir / dest_name
            counter += 1

        # Copy notebook file directly (scicode-lint handles .ipynb)
        shutil.copy2(nb_file, dest_path)

        # Detect imports from extracted code
        ml_imports, scientific_imports = detect_imports(content)

        # Create record
        record = {
            "file_path": f"files/{dest_name}",
            "original_path": str(nb_file.relative_to(raw_dir)),
            "repo_name": REPO_NAME,
            "repo_url": REPO_URL,
            "data_source": "leakage_paper",
            "paper_url": PAPER_URL,
            "paper_title": PAPER_TITLE,
            "domain": "data_science",
            "is_notebook": "True",
            "file_size": dest_path.stat().st_size,
            "line_count": content.count("\n") + 1,
            "ml_imports": ";".join(ml_imports),
            "scientific_imports": ";".join(scientific_imports),
        }
        records.append(record)

    # Write manifest
    if records:
        fieldnames = list(records[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Generated manifest: {manifest_path} ({len(records)} files)")
    else:
        logger.warning("No files to include in manifest")

    return manifest_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download and prepare leakage paper sample notebooks"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download sample-notebooks.zip and ground truth",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Generate manifest from downloaded data",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run full pipeline (download + prepare)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/re-extract",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit number of files in manifest",
    )
    args = parser.parse_args()

    # Default to showing help if no action specified
    if not (args.download or args.prepare or args.run):
        parser.print_help()
        return

    if args.download or args.run:
        download_data(force=args.force)
        download_ground_truth(force=args.force)

    if args.prepare or args.run:
        manifest_path = prepare_manifest(max_files=args.max_files)
        logger.info("=" * 50)
        logger.info("To run analysis:")
        logger.info("  python -m real_world_demo.run_analysis \\")
        logger.info(f"    --manifest {manifest_path} \\")
        logger.info(f"    --base-dir {LEAKAGE_PAPER_COLLECTED_DIR}")


if __name__ == "__main__":
    main()
