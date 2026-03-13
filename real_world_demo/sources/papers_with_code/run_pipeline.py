"""Main pipeline orchestrator for real-world demo.

Runs all stages of the pipeline: filter papers, clone repos, filter files,
generate manifest, and run analysis.

Usage:
    python run_pipeline.py --papers 100           # Limit to 100 papers
    python run_pipeline.py --stage clone         # Run only clone stage
    python run_pipeline.py --stage all --force   # Full run, force all stages
"""

import argparse
import subprocess
import sys

from loguru import logger

from real_world_demo.config import CLONED_DIR, COLLECTED_DIR, DATA_DIR, REPORTS_DIR


def run_stage(
    stage_name: str,
    module: str,
    args: list[str],
    check: bool = True,
) -> bool:
    """Run a pipeline stage as a subprocess.

    Args:
        stage_name: Human-readable stage name for logging.
        module: Module to run (e.g., 'real_world_demo.filter_papers').
        args: Command-line arguments to pass.
        check: Whether to raise on non-zero exit code.

    Returns:
        True if stage succeeded, False otherwise.
    """
    logger.info("=" * 60)
    logger.info(f"STAGE: {stage_name}")
    logger.info("=" * 60)

    cmd = [sys.executable, "-m", module] + args
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=check)
        if result.returncode == 0:
            logger.info(f"STAGE {stage_name}: COMPLETED")
            return True
        else:
            logger.error(f"STAGE {stage_name}: FAILED (exit code {result.returncode})")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"STAGE {stage_name}: FAILED ({e})")
        return False
    except Exception as e:
        logger.error(f"STAGE {stage_name}: ERROR ({e})")
        return False


def check_prerequisites(stage: str) -> bool:
    """Check if prerequisites for a stage are met.

    Args:
        stage: Stage name to check.

    Returns:
        True if prerequisites are met.
    """
    prerequisites = {
        "abstract_filter": DATA_DIR / "filtered_papers.json",
        "clone": DATA_DIR / "ai_science_papers.json",
        "files": CLONED_DIR,
        "prefilter": DATA_DIR / "qualifying_files.json",
        "manifest": DATA_DIR / "pipeline_files.json",
        "analyze": COLLECTED_DIR / "manifest.csv",
    }

    if stage not in prerequisites:
        return True

    prereq = prerequisites[stage]
    if not prereq.exists():
        logger.error(f"Prerequisite not found for stage '{stage}': {prereq}")
        logger.error("Run the previous stage first.")
        return False

    return True


def main() -> None:
    """Main entry point for pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Real-world demo pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m real_world_demo.sources.papers_with_code.run_pipeline --papers 100
      Run with 100 papers (all stages)

  python -m real_world_demo.sources.papers_with_code.run_pipeline --stage clone
      Run only the clone stage (requires filter stage completed)

  python -m real_world_demo.sources.papers_with_code.run_pipeline --stage all --force
      Run all stages, forcing regeneration of existing outputs

  python -m real_world_demo.sources.papers_with_code.run_pipeline \\
      --stage filter --domains biology,medical
      Filter only biology and medical domains
        """,
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "filter",
            "abstract_filter",
            "clone",
            "files",
            "prefilter",
            "manifest",
            "analyze",
        ],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--papers",
        type=int,
        help="Number of papers to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running stages even if output exists",
    )
    parser.add_argument(
        "--domains",
        type=str,
        help="Comma-separated list of domains to include (filter stage only)",
    )
    parser.add_argument(
        "--max-concurrent-clones",
        type=int,
        default=3,
        help="Max concurrent clone operations (clone stage only, default: 3)",
    )
    parser.add_argument(
        "--max-concurrent-analysis",
        type=int,
        default=5,
        help="Max concurrent analysis operations (analyze stage only, default: 5)",
    )
    parser.add_argument(
        "--max-files-analyze",
        type=int,
        help="Limit number of files to analyze (analyze stage only)",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="Disable balanced sampling across domains (filter stage only)",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:5001",
        help="vLLM server URL (default: http://localhost:5001)",
    )
    parser.add_argument(
        "--max-concurrent-prefilter",
        type=int,
        default=10,
        help="Max concurrent prefilter requests (prefilter stage only, default: 10)",
    )
    args = parser.parse_args()

    logger.info("Real-World Demo Pipeline")
    logger.info(f"Stage: {args.stage}")
    if args.papers:
        logger.info(f"Papers: {args.papers}")

    stages_to_run = (
        ["filter", "abstract_filter", "clone", "files", "prefilter", "manifest", "analyze"]
        if args.stage == "all"
        else [args.stage]
    )

    for stage in stages_to_run:
        # Check prerequisites
        if not check_prerequisites(stage):
            logger.error(f"Stopping pipeline at stage '{stage}' due to missing prerequisites")
            sys.exit(1)

        # Build stage-specific arguments
        stage_args = []
        if args.force:
            stage_args.append("--force")

        if stage == "filter":
            if args.papers:
                stage_args.extend(["--limit", str(args.papers)])
            if args.domains:
                stage_args.extend(["--domains", args.domains])
            if args.no_balanced:
                stage_args.append("--no-balanced")

        elif stage == "abstract_filter":
            stage_args.extend(["--max-concurrent", str(args.max_concurrent_prefilter)])

        elif stage == "clone":
            if args.papers:
                stage_args.extend(["--papers", str(args.papers)])
            stage_args.extend(["--max-concurrent", str(args.max_concurrent_clones)])

        elif stage == "files":
            pass  # No special args

        elif stage == "prefilter":
            stage_args.extend(["--llm-url", args.llm_url])
            stage_args.extend(["--max-concurrent", str(args.max_concurrent_prefilter)])

        elif stage == "manifest":
            pass  # No special args

        elif stage == "analyze":
            stage_args.extend(["--max-concurrent", str(args.max_concurrent_analysis)])
            if args.max_files_analyze:
                stage_args.extend(["--max-files", str(args.max_files_analyze)])

        # Run stage
        stage_to_module = {
            "filter": "real_world_demo.sources.papers_with_code.filter_papers",
            "abstract_filter": "real_world_demo.sources.papers_with_code.filter_abstracts",
            "clone": "real_world_demo.sources.papers_with_code.clone_repos",
            "files": "real_world_demo.sources.papers_with_code.filter_files",
            "prefilter": "real_world_demo.sources.papers_with_code.prefilter_files",
            "manifest": "real_world_demo.sources.papers_with_code.generate_manifest",
            "analyze": "real_world_demo.run_analysis",
        }
        module_name = stage_to_module[stage]

        success = run_stage(stage.upper(), module_name, stage_args)

        if not success:
            logger.error(f"Pipeline failed at stage '{stage}'")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    # Print output locations
    logger.info("Output locations:")
    logger.info(f"  Filtered papers: {DATA_DIR / 'filtered_papers.json'}")
    logger.info(f"  AI+Science papers: {DATA_DIR / 'ai_science_papers.json'}")
    logger.info(f"  AI+Science papers: {DATA_DIR / 'ai_science_papers.json'}")
    logger.info(f"  Clone results: {DATA_DIR / 'clone_results.json'}")
    logger.info(f"  Qualifying files: {DATA_DIR / 'qualifying_files.json'}")
    logger.info(f"  Manifest: {COLLECTED_DIR / 'manifest.csv'}")
    logger.info(f"  Findings report: {REPORTS_DIR / 'findings_summary.md'}")


if __name__ == "__main__":
    main()
