#!/usr/bin/env python3
"""Experiment: Test different max_completion_tokens values.

Runs full eval suite with different token limits to find optimal value
that balances accuracy vs execution time.

Usage:
    python benchmarks/max_tokens_experiment.py

Results saved to: benchmarks/reports/max_tokens/comparison_YYYYMMDD_HHMMSS.json
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

TOKEN_VALUES = [16384, 8192, 6144, 4096, 2048, 1024, 512]
REPO_ROOT = Path(__file__).parent.parent


def run_eval(max_tokens: int) -> tuple[dict[str, Any], float]:
    """Run eval with specific max_completion_tokens, return metrics and time."""
    env = os.environ.copy()
    env["SCICODE_LINT_MAX_COMPLETION_TOKENS"] = str(max_tokens)

    start = time.time()
    subprocess.run(
        ["python", "evals/run_eval.py"],
        env=env,
        cwd=REPO_ROOT,
        check=True,
    )
    elapsed = time.time() - start

    with open(REPO_ROOT / "evals/reports/judge/llm_judge_report.json") as f:
        return json.load(f), elapsed


def main() -> None:
    """Run experiment with all token values and save comparison results."""
    results: dict[str, Any] = {
        "experiment": "max_output_tokens",
        "timestamp": datetime.now().isoformat(),
        "configs": {},
    }

    for tokens in TOKEN_VALUES:
        print(f"\n{'=' * 60}\nRunning with max_completion_tokens={tokens}\n{'=' * 60}")
        metrics, elapsed = run_eval(tokens)
        results["configs"][str(tokens)] = {
            "overall_accuracy": metrics["overall_accuracy"],
            "positive_accuracy": metrics["positive_accuracy"],
            "negative_accuracy": metrics["negative_accuracy"],
            "execution_time_seconds": round(elapsed, 1),
        }
        print(f"Done: {metrics['overall_accuracy']:.2%} accuracy in {elapsed:.0f}s")

    # Save results
    out_dir = REPO_ROOT / "benchmarks/reports/max_tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
