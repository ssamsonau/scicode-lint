#!/usr/bin/env python3
"""Benchmark scan time vs file size.

Measures per-file scan timing across different file sizes to establish
performance characteristics and validate documentation claims.

Two modes:
- Full scan (all 66 patterns): measures total scan time per file size
- Single pattern: measures per-pattern cost, isolating LLM call overhead from pattern count

Usage:
    python benchmarks/file_size_benchmark.py                    # full scan, 2 runs
    python benchmarks/file_size_benchmark.py --runs 3 --warmup  # 3 runs with warmup
    python benchmarks/file_size_benchmark.py --single-pattern ml-001  # single pattern mode

Results saved to: benchmarks/reports/file_size/benchmark_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Ordered by size
FIXTURE_FILES = [
    "small_30_lines.py",
    "medium_200_lines.py",
    "large_500_lines.py",
    "xlarge_1000_lines.py",
]


def count_lines(filepath: Path) -> int:
    """Count lines in a file."""
    return sum(1 for _ in filepath.open())


def run_scan(filepath: Path, pattern: str | None = None) -> tuple[float, int, str]:
    """Run scicode-lint lint on a file and return (elapsed_seconds, exit_code, output).

    Args:
        filepath: Path to the file to scan.
        pattern: Optional pattern ID to run (e.g. 'ml-001'). None = all patterns.
    """
    cmd = ["scicode-lint", "lint", str(filepath)]
    if pattern:
        cmd.extend(["--pattern", pattern])

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    elapsed = time.time() - start
    output = result.stdout + result.stderr
    return elapsed, result.returncode, output


def count_findings(output: str) -> int:
    """Extract number of findings from scicode-lint output."""
    for line in output.split("\n"):
        if "issues found" in line:
            parts = line.split("—")
            if len(parts) >= 2:
                count_str = parts[1].strip().split()[0]
                try:
                    return int(count_str)
                except ValueError:
                    pass
        if "No issues found" in line:
            return 0
    return -1


def run_benchmark(n_runs: int, do_warmup: bool, pattern: str | None = None) -> dict[str, Any]:
    """Run the benchmark suite.

    Args:
        n_runs: Number of runs per file.
        do_warmup: Whether to run a warmup scan first.
        pattern: Optional single pattern ID. None = full scan.
    """
    mode = f"single pattern ({pattern})" if pattern else "full scan (all patterns)"
    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "pattern": pattern,
        "n_runs": n_runs,
        "files": [],
    }

    if do_warmup:
        print("Running warmup scan (populating prefix cache)...")
        warmup_file = FIXTURES_DIR / FIXTURE_FILES[0]
        run_scan(warmup_file, pattern=pattern)
        print("  Warmup complete.\n")

    for filename in FIXTURE_FILES:
        filepath = FIXTURES_DIR / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} not found")
            continue

        n_lines = count_lines(filepath)
        print(f"Benchmarking: {filename} ({n_lines} lines) [{mode}]")

        timings = []
        findings = -1

        for run_idx in range(n_runs):
            elapsed, exit_code, output = run_scan(filepath, pattern=pattern)
            timings.append(elapsed)

            if findings == -1:
                findings = count_findings(output)

            print(f"  Run {run_idx + 1}/{n_runs}: {elapsed:.1f}s")

        file_result = {
            "filename": filename,
            "lines": n_lines,
            "findings": findings,
            "timings_seconds": [round(t, 2) for t in timings],
            "mean_seconds": round(sum(timings) / len(timings), 2),
            "min_seconds": round(min(timings), 2),
            "max_seconds": round(max(timings), 2),
        }

        if len(timings) > 1:
            spread = max(timings) - min(timings)
            file_result["spread_seconds"] = round(spread, 2)

        results["files"].append(file_result)
        print(
            f"  → Mean: {file_result['mean_seconds']}s "
            f"(min={file_result['min_seconds']}s, max={file_result['max_seconds']}s)\n"
        )

    return results


def format_report(results: dict[str, Any]) -> str:
    """Format results as markdown."""
    mode = results.get("mode", "full scan")
    lines = [
        "# File Size Benchmark Report",
        "",
        f"**Date:** {results['timestamp'][:10]}",
        f"**Mode:** {mode}",
        f"**Runs per file:** {results['n_runs']}",
        "",
        "## Results",
        "",
        "| File | Lines | Findings | Mean (s) | Min (s) | Max (s) | Spread (s) |",
        "|------|------:|--------:|---------:|--------:|--------:|-----------:|",
    ]

    for f in results["files"]:
        spread = f.get("spread_seconds", "-")
        lines.append(
            f"| {f['filename']} | {f['lines']} | {f['findings']} | "
            f"{f['mean_seconds']} | {f['min_seconds']} | {f['max_seconds']} | {spread} |"
        )

    if len(results["files"]) >= 2:
        smallest = results["files"][0]
        largest = results["files"][-1]
        ratio = (
            largest["mean_seconds"] / smallest["mean_seconds"]
            if smallest["mean_seconds"] > 0
            else 0
        )
        line_ratio = largest["lines"] / smallest["lines"] if smallest["lines"] > 0 else 0

        if ratio < line_ratio * 0.8:
            scaling = "Sub-linear"
        elif ratio < line_ratio * 1.2:
            scaling = "Roughly linear"
        else:
            scaling = "Super-linear"

        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- **Smallest file:** {smallest['lines']} lines → {smallest['mean_seconds']}s",
                f"- **Largest file:** {largest['lines']} lines → {largest['mean_seconds']}s",
                f"- **Line count ratio:** {line_ratio:.0f}x",
                f"- **Time ratio:** {ratio:.1f}x",
                f"- **Scaling:** {scaling} ({ratio:.1f}x time for {line_ratio:.0f}x lines)",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Run file size benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark scan time vs file size")
    parser.add_argument("--runs", type=int, default=2, help="Runs per file (default: 2)")
    parser.add_argument("--warmup", action="store_true", help="Run warmup scan first")
    parser.add_argument(
        "--single-pattern",
        type=str,
        default=None,
        help="Run only this pattern (e.g. ml-001). Default: all patterns.",
    )
    args = parser.parse_args()

    # Verify fixtures exist
    missing = [f for f in FIXTURE_FILES if not (FIXTURES_DIR / f).exists()]
    if missing:
        print(f"Error: Missing fixture files: {missing}")
        print(f"Expected in: {FIXTURES_DIR}")
        return

    mode_label = f"single pattern ({args.single_pattern})" if args.single_pattern else "full scan"
    print(f"File Size Benchmark: {len(FIXTURE_FILES)} files × {args.runs} runs [{mode_label}]\n")

    results = run_benchmark(args.runs, args.warmup, pattern=args.single_pattern)

    # Save results
    out_dir = REPO_ROOT / "benchmarks/reports/file_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.single_pattern}" if args.single_pattern else "_full"
    json_file = out_dir / f"benchmark_{ts}{suffix}.json"
    md_file = out_dir / "BENCHMARK_SUMMARY.md"

    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    with open(md_file, "w") as f:
        f.write(format_report(results))

    print("=" * 60)
    print(format_report(results))
    print("=" * 60)
    print(f"\nJSON: {json_file}")
    print(f"Summary: {md_file}")


if __name__ == "__main__":
    main()
