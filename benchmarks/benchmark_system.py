#!/usr/bin/env python3
"""
Benchmarking script for scicode-lint.

Run this to measure performance on your machine and save results locally.
Results are saved to benchmark_results.json (not committed to git).
"""

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


def get_system_info() -> dict[str, Any]:
    """Collect system information."""
    info: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }

    # Try to get CPU info on Linux
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    # Try to get memory info on Linux
    if sys.platform == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["total_memory_gb"] = round(kb / 1024 / 1024, 1)
                        break
        except Exception:
            pass

    return info


def check_llm_backend() -> dict[str, Any]:
    """Check which LLM backend is available and get info."""
    import httpx

    backend_info: dict[str, Any] = {"available": False, "backend": None, "url": None}

    # Try vLLM
    for url in ["http://localhost:5001", "http://localhost:8000"]:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{url}/v1/models")
                if response.status_code == 200:
                    backend_info["available"] = True
                    backend_info["backend"] = "vLLM"
                    backend_info["url"] = url
                    backend_info["models"] = response.json()
                    return backend_info
        except Exception:
            continue

    return backend_info


def run_benchmark_test(test_files: list[Path], model: str | None = None) -> dict[str, Any]:
    """Run benchmark on test files."""
    logger.info(f"Running benchmark on {len(test_files)} files...")

    # Build command
    cmd = ["scicode-lint", "check"] + [str(f) for f in test_files]
    if model:
        cmd.extend(["--model", model])

    # Run with timing
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        elapsed = time.time() - start_time

        return {
            "success": True,
            "elapsed_seconds": round(elapsed, 2),
            "files_count": len(test_files),
            "avg_time_per_file": round(elapsed / len(test_files), 2),
            "files_per_minute": round(len(test_files) / (elapsed / 60), 1),
            "return_code": result.returncode,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": "Timeout (5 minutes)",
            "elapsed_seconds": round(elapsed, 2),
            "files_count": len(test_files),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "files_count": len(test_files),
        }


def main() -> None:
    """Run benchmarks and save results."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    logger.info("=" * 60)
    logger.info("scicode-lint Benchmark")
    logger.info("=" * 60)

    # Collect system info
    logger.info("Collecting system information...")
    system_info = get_system_info()
    logger.info(f"Platform: {system_info['platform']}")
    logger.info(f"Python: {system_info['python_version']}")
    if "cpu_model" in system_info:
        logger.info(f"CPU: {system_info['cpu_model']}")
    if "total_memory_gb" in system_info:
        logger.info(f"Memory: {system_info['total_memory_gb']} GB")

    # Check LLM backend
    logger.info("\nChecking LLM backend...")
    backend_info = check_llm_backend()
    if backend_info["available"]:
        logger.info(f"Backend: {backend_info['backend']} at {backend_info['url']}")
        logger.info(f"Models: {backend_info['models']}")
    else:
        logger.error("No vLLM backend available!")
        logger.error("Please start vLLM before running benchmark:")
        logger.error("  vllm serve --model RedHatAI/gemma-3-12b-it-FP8-dynamic")
        sys.exit(1)

    # Find test files
    logger.info("\nFinding test files...")
    repo_root = Path(__file__).parent
    test_files = list((repo_root / "specs" / "eval").rglob("*.py"))

    if not test_files:
        # Fallback to src files if no test files
        test_files = list((repo_root / "src").rglob("*.py"))

    if not test_files:
        logger.error("No Python files found to benchmark!")
        sys.exit(1)

    # Limit to reasonable number for benchmarking
    if len(test_files) > 20:
        logger.info(f"Found {len(test_files)} files, using first 20 for benchmark")
        test_files = test_files[:20]
    else:
        logger.info(f"Found {len(test_files)} files")

    # Run benchmark
    logger.info("\n" + "=" * 60)
    logger.info("Running benchmark...")
    logger.info("=" * 60)

    benchmark_results = run_benchmark_test(test_files)

    # Compile full results
    results = {
        "system": system_info,
        "llm_backend": backend_info,
        "benchmark": benchmark_results,
    }

    # Save results to JSON
    output_file = repo_root / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)

    if benchmark_results["success"]:
        logger.info("✓ Benchmark completed successfully")
        logger.info(f"Files checked: {benchmark_results['files_count']}")
        logger.info(f"Total time: {benchmark_results['elapsed_seconds']}s")
        logger.info(f"Average time per file: {benchmark_results['avg_time_per_file']}s")
        logger.info(f"Throughput: {benchmark_results['files_per_minute']} files/minute")
    else:
        logger.error(f"✗ Benchmark failed: {benchmark_results.get('error', 'Unknown error')}")

    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)

    # Also save human-readable text version
    text_output = output_file.with_suffix(".txt")
    with open(text_output, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("scicode-lint Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {system_info['timestamp']}\n\n")

        f.write("SYSTEM INFO\n")
        f.write("-" * 60 + "\n")
        for key, value in system_info.items():
            if key != "timestamp":
                f.write(f"{key}: {value}\n")

        f.write("\nLLM BACKEND\n")
        f.write("-" * 60 + "\n")
        f.write(f"Backend: {backend_info.get('backend', 'N/A')}\n")
        f.write(f"URL: {backend_info.get('url', 'N/A')}\n")
        f.write(f"Models: {backend_info.get('models', 'N/A')}\n")

        f.write("\nBENCHMARK RESULTS\n")
        f.write("-" * 60 + "\n")
        if benchmark_results["success"]:
            f.write("Status: SUCCESS\n")
            f.write(f"Files checked: {benchmark_results['files_count']}\n")
            f.write(f"Total time: {benchmark_results['elapsed_seconds']}s\n")
            f.write(f"Average per file: {benchmark_results['avg_time_per_file']}s\n")
            f.write(f"Throughput: {benchmark_results['files_per_minute']} files/min\n")
        else:
            f.write("Status: FAILED\n")
            f.write(f"Error: {benchmark_results.get('error', 'Unknown')}\n")

    logger.info(f"Human-readable results saved to: {text_output}")


if __name__ == "__main__":
    main()
