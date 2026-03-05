"""Command-line interface."""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

from scicode_lint.config import LinterConfig, LLMConfig, Severity
from scicode_lint.linter import SciCodeLinter
from scicode_lint.output.formatter import LintError, LintResult, format_findings


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-powered linter for scientific Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        choices=["check"],
        help="Command to run",
    )

    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to check",
    )

    parser.add_argument(
        "--severity",
        type=str,
        help="Comma-separated severity levels to check (critical,high,medium)",
        default="critical,high,medium",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (text for humans, json for GenAI agents)",
    )

    parser.add_argument(
        "--json-errors",
        action="store_true",
        help="Include errors in JSON output (useful for GenAI agents)",
    )

    parser.add_argument(
        "--vllm-url",
        type=str,
        help="vLLM server URL (default: auto-detect on ports 5001 or 8000)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: auto-detect from vLLM server)",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (0.0-1.0)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        help=(
            "Comma-separated pattern IDs to check (e.g., ml-001,ml-002). "
            "If not specified, all patterns are checked."
        ),
    )

    parser.add_argument(
        "--category",
        type=str,
        help=(
            "Comma-separated categories to check (e.g., ml-correctness,pytorch-training). "
            "If not specified, all categories are checked."
        ),
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -v, -vv, or -vvv)",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode (shows detailed timing statistics)",
    )

    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    args = parse_args(argv)

    # Configure logging based on verbosity
    logger.remove()  # Remove default handler
    if args.verbose == 0:
        # Only show warnings and errors to stderr
        logger.add(sys.stderr, level="WARNING", format="<level>{message}</level>")
    elif args.verbose == 1:
        # Show info messages (includes timing)
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )
    elif args.verbose == 2:
        # Show debug messages
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
        )
    else:  # args.verbose >= 3
        # Show everything with full context
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}"
            ),
        )

    start_time = time.time()

    # Parse severity levels
    severity_map = {
        "critical": Severity.CRITICAL,
        "high": Severity.HIGH,
        "medium": Severity.MEDIUM,
    }

    enabled_severities = set()
    for s in args.severity.split(","):
        s = s.strip().lower()
        if s in severity_map:
            enabled_severities.add(severity_map[s])

    # Parse pattern filter
    enabled_patterns = None
    if args.pattern:
        enabled_patterns = {p.strip() for p in args.pattern.split(",")}
        logger.info(f"Filtering to patterns: {enabled_patterns}")

    # Parse category filter
    enabled_categories = None
    if args.category:
        enabled_categories = {c.strip() for c in args.category.split(",")}
        logger.info(f"Filtering to categories: {enabled_categories}")

    # Configure LLM (auto-detects vLLM if no URL specified)
    llm_config = LLMConfig(
        base_url=args.vllm_url or "",
        model=args.model or "",
    )

    # Configure linter
    linter_config = LinterConfig(
        # patterns_dir will use default from get_default_patterns_dir()
        llm_config=llm_config,
        min_confidence=args.min_confidence,
        enabled_severities=enabled_severities,
        enabled_patterns=enabled_patterns,
        enabled_categories=enabled_categories,
    )

    # Create linter
    linter = SciCodeLinter(linter_config)

    # Collect files to check
    files_to_check: list[Path] = []
    for path in args.paths:
        if path.is_file():
            if path.suffix in {".py", ".ipynb"}:
                files_to_check.append(path)
        elif path.is_dir():
            # Recursively find Python files
            files_to_check.extend(path.rglob("*.py"))
            files_to_check.extend(path.rglob("*.ipynb"))

    if not files_to_check:
        logger.error("No Python files found to check.")
        return 1

    logger.info(f"Found {len(files_to_check)} files to check")

    # Check all files
    results = []
    for idx, file_path in enumerate(files_to_check, 1):
        logger.info(f"Processing file {idx}/{len(files_to_check)}")
        try:
            result = linter.check_file(file_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Error checking {file_path}: {e}")
            # If --json-errors is enabled, capture errors in structured format
            if args.json_errors:
                error_details = None
                # Try to extract structured details from exception if available
                if hasattr(e, "to_dict"):
                    error_details = e.to_dict()

                error = LintError(
                    file=file_path,
                    error_type=type(e).__name__,
                    message=str(e),
                    details=error_details,
                )
                results.append(LintResult(file=file_path, findings=[], error=error))
            continue

    # Calculate timing statistics
    elapsed = time.time() - start_time
    total_findings = sum(len(r.findings) for r in results)

    # Format and print results
    output = format_findings(results, output_format=args.format)
    print(output)

    # Show benchmark/timing summary
    if args.benchmark or args.verbose > 0:
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files checked: {len(results)}")
        logger.info(f"Total findings: {total_findings}")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Average time per file: {elapsed / len(results):.2f}s")
        logger.info(f"Files per minute: {len(results) / (elapsed / 60):.1f}")

        # Show per-file breakdown in benchmark mode
        if args.benchmark and results:
            logger.info("\nPer-file breakdown:")
            for result in results:
                logger.info(f"  {result.file.name}: {len(result.findings)} findings")

        logger.info("=" * 60)

    logger.success(f"Linting completed in {elapsed:.2f}s")

    # Return non-zero if any findings
    return 1 if total_findings > 0 else 0
