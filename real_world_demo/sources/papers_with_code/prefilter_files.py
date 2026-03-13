"""Prefilter files using vLLM to identify ML pipeline code.

Runs a quick LLM check to filter out utility/visualization-only files
before expensive full analysis.

Usage:
    python prefilter_files.py [--max-concurrent 10]
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from real_world_demo.config import DATA_DIR
from scicode_lint.config import load_llm_config

# Prefilter prompt - designed to be fast and focused
# Covers all scicode-lint categories: ai-training, ai-inference,
# scientific-numerical, scientific-performance, scientific-reproducibility
PREFILTER_PROMPT = """Analyze this Python code. Does it contain scientific/ML code worth linting?

INCLUDE (answer YES) - code with ANY of:
- ML training: model fitting, train/test split, cross-validation, hyperparameter tuning
- ML inference: model prediction, batch processing, deployment code
- Data preprocessing: scaling, normalization, encoding, feature engineering
- Scientific computation: numerical algorithms, simulations, statistical analysis
- Array/matrix operations: numpy/scipy intensive computation
- Random number usage: sampling, stochastic algorithms, Monte Carlo
- Performance-critical code: GPU operations, parallel processing, large data

EXCLUDE (answer NO):
- Pure visualization/plotting only
- Simple utility functions (logging, file I/O helpers)
- Configuration files, setup.py, CLI argument parsing only
- Documentation generation, test fixtures
- Pure data download without processing

CODE:
```python
{code}
```

Is this scientific/ML code worth linting? Answer only YES or NO."""


async def check_file_with_llm(
    file_path: Path,
    semaphore: asyncio.Semaphore,
    llm_base_url: str,
    model_name: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Check if a file contains ML pipeline code using LLM.

    Args:
        file_path: Path to Python file.
        semaphore: Semaphore for concurrency control.
        llm_base_url: Base URL for vLLM server.
        model_name: Model name for API calls.
        timeout: Request timeout in seconds.

    Returns:
        Dict with file_path, is_pipeline, and raw_response.
    """
    import httpx

    async with semaphore:
        try:
            # Read file content
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Cannot read {file_path}: {e}")
                return {
                    "file_path": str(file_path),
                    "is_pipeline": False,
                    "error": f"read_error: {e}",
                }

            # Truncate very long files (keep first ~4000 chars for speed)
            if len(content) > 4000:
                content = content[:4000] + "\n... [truncated]"

            prompt = PREFILTER_PROMPT.format(code=content)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{llm_base_url}/v1/completions",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "max_tokens": 10,
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                result = response.json()

            answer = result["choices"][0]["text"].strip().upper()
            is_pipeline = "YES" in answer

            logger.debug(f"{file_path.name}: {'YES' if is_pipeline else 'NO'}")
            return {
                "file_path": str(file_path),
                "is_pipeline": is_pipeline,
                "raw_response": answer,
            }

        except TimeoutError:
            logger.warning(f"Timeout checking {file_path}")
            return {
                "file_path": str(file_path),
                "is_pipeline": True,  # Include on timeout (conservative)
                "error": "timeout",
            }
        except Exception as e:
            logger.warning(f"Error checking {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "is_pipeline": True,  # Include on error (conservative)
                "error": str(e),
            }


async def prefilter_all_files(
    qualifying_files: list[dict[str, Any]],
    llm_base_url: str,
    model_name: str,
    max_concurrent: int = 10,
    timeout: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prefilter all qualifying files using LLM.

    Args:
        qualifying_files: List of file records from filter_files.
        llm_base_url: Base URL for vLLM server.
        model_name: Model name for API calls.
        max_concurrent: Maximum concurrent LLM requests.
        timeout: Request timeout per file.

    Returns:
        Tuple of (pipeline_files, filtered_out_files).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for file_info in qualifying_files:
        file_path = Path(file_info["file_path"])
        tasks.append(check_file_with_llm(file_path, semaphore, llm_base_url, model_name, timeout))

    logger.info(f"Prefiltering {len(tasks)} files with LLM...")
    results = await asyncio.gather(*tasks)

    # Partition into pipeline and non-pipeline files
    pipeline_files = []
    filtered_out = []

    for file_info, result in zip(qualifying_files, results, strict=True):
        if result.get("is_pipeline", False):
            # Add prefilter result to file_info
            file_info["prefilter_response"] = result.get("raw_response", "")
            pipeline_files.append(file_info)
        else:
            file_info["prefilter_response"] = result.get("raw_response", "")
            filtered_out.append(file_info)

    return pipeline_files, filtered_out


def load_qualifying_files(input_file: Path) -> list[dict[str, Any]]:
    """Load qualifying files from JSON.

    Args:
        input_file: Path to qualifying_files.json.

    Returns:
        List of file records.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Qualifying files not found: {input_file}. Run filter_files.py first."
        )

    with open(input_file) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def save_results(
    pipeline_files: list[dict[str, Any]],
    filtered_out: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save prefilter results.

    Args:
        pipeline_files: Files identified as ML pipeline code.
        filtered_out: Files filtered out.
        output_dir: Output directory.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save pipeline files (these will be analyzed)
    pipeline_file = output_dir / "pipeline_files.json"
    with open(pipeline_file, "w") as f:
        json.dump(pipeline_files, f, indent=2)
    logger.info(f"Saved {len(pipeline_files)} pipeline files to {pipeline_file}")

    # Save filtered out files (for review/debugging)
    filtered_file = output_dir / "prefilter_excluded.json"
    with open(filtered_file, "w") as f:
        json.dump(filtered_out, f, indent=2)
    logger.info(f"Saved {len(filtered_out)} excluded files to {filtered_file}")


def print_summary(pipeline_files: list[dict[str, Any]], filtered_out: list[dict[str, Any]]) -> None:
    """Print prefilter summary.

    Args:
        pipeline_files: Files kept.
        filtered_out: Files excluded.
    """
    total = len(pipeline_files) + len(filtered_out)
    kept = len(pipeline_files)
    excluded = len(filtered_out)

    logger.info("=" * 50)
    logger.info("Prefilter Summary:")
    logger.info(f"  Total files checked: {total}")
    logger.info(f"  ML pipeline files: {kept} ({100 * kept / total:.1f}%)")
    logger.info(f"  Excluded (non-pipeline): {excluded} ({100 * excluded / total:.1f}%)")


def get_llm_defaults() -> tuple[str, str, int, int]:
    """Get LLM defaults from scicode_lint config.

    Returns:
        Tuple of (base_url, model_name, timeout, max_concurrent).
    """
    config = load_llm_config()

    # Use config values (these come from config.toml)
    base_url = config.base_url or "http://localhost:5001"
    model_name = config.model_served_name or config.model
    timeout = config.timeout
    # Default concurrent - could be added to config.toml if needed
    max_concurrent = 10

    return base_url, model_name, timeout, max_concurrent


def main() -> None:
    """Main entry point for prefiltering."""
    # Get defaults from scicode_lint config
    default_url, default_model, default_timeout, default_concurrent = get_llm_defaults()

    parser = argparse.ArgumentParser(
        description="Prefilter files using LLM to identify ML pipeline code"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DATA_DIR / "qualifying_files.json",
        help="Input file with qualifying files list",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=default_url,
        help=f"vLLM server URL (default: {default_url})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Model name for API calls (default: {default_model})",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=default_concurrent,
        help=f"Maximum concurrent LLM requests (default: {default_concurrent})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=default_timeout,
        help=f"Request timeout per file in seconds (default: {default_timeout})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-filtering even if output exists",
    )
    args = parser.parse_args()

    # Check if output already exists
    pipeline_file = args.output_dir / "pipeline_files.json"
    if pipeline_file.exists() and not args.force:
        logger.info(f"Output already exists: {pipeline_file}")
        logger.info("Use --force to re-filter")
        with open(pipeline_file) as f:
            existing = json.load(f)
        logger.info(f"Existing pipeline files: {len(existing)}")
        return

    # Load qualifying files
    qualifying_files = load_qualifying_files(args.input_file)
    logger.info(f"Loaded {len(qualifying_files)} qualifying files")

    if not qualifying_files:
        logger.warning("No files to prefilter!")
        return

    logger.info(f"Using LLM: {args.llm_url} (model: {args.model})")

    # Run prefilter
    pipeline_files, filtered_out = asyncio.run(
        prefilter_all_files(
            qualifying_files,
            args.llm_url,
            args.model,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
        )
    )

    # Save results
    save_results(pipeline_files, filtered_out, args.output_dir)

    # Print summary
    print_summary(pipeline_files, filtered_out)


if __name__ == "__main__":
    main()
