#!/usr/bin/env python3
"""
Simple benchmark: Sequential vs Concurrent.
No semaphores, just pure comparison.
"""

import asyncio
import sys
import time

from loguru import logger

from scicode_lint.config import LLMConfig
from scicode_lint.detectors.catalog import DetectionCatalog
from scicode_lint.detectors.prompts import generate_detection_prompt, get_system_prompt
from scicode_lint.llm.client import create_client
from scicode_lint.llm.models import DetectionResult

logger.remove()
logger.add(
    sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>"
)


async def run_sequential(patterns, code, llm, system_prompt):
    """Run patterns one at a time."""
    results = []
    for pattern in patterns:
        user_prompt = generate_detection_prompt(code, pattern)
        result = await llm.async_complete_structured(system_prompt, user_prompt, DetectionResult)
        results.append(result)
    return results


async def run_concurrent(patterns, code, llm, system_prompt):
    """Run all patterns at once."""
    prompts = [generate_detection_prompt(code, pattern) for pattern in patterns]
    tasks = [llm.async_complete_structured(system_prompt, p, DetectionResult) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results


async def main():
    logger.info("=" * 70)
    logger.info("SIMPLE BENCHMARK: Sequential vs Concurrent")
    logger.info("=" * 70)

    # Setup
    code = """import torch
def train(model, data):
    optimizer = torch.optim.Adam(model.parameters())
    for batch in data:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        # BUG: missing zero_grad()
    return model
"""

    catalog = DetectionCatalog()
    patterns = list(catalog.patterns)[:5]  # First 5 patterns

    config = LLMConfig(base_url="http://localhost:5001", temperature=0.0)
    llm = create_client(config)
    system_prompt = get_system_prompt()

    logger.info(f"Testing {len(patterns)} patterns: {', '.join(p.id for p in patterns)}")
    logger.info("")

    # Test 1: Sequential
    logger.info("Test 1: Sequential (one at a time)")
    start = time.time()
    results_seq = await run_sequential(patterns, code, llm, system_prompt)
    time_seq = time.time() - start
    findings_seq = sum(1 for r in results_seq if r.detected and r.confidence >= 0.7)
    logger.success(f"Sequential: {time_seq:.1f}s | {findings_seq} findings")

    await asyncio.sleep(2)

    # Test 2: Concurrent
    logger.info("")
    logger.info("Test 2: Concurrent (all at once)")
    start = time.time()
    results_conc = await run_concurrent(patterns, code, llm, system_prompt)
    time_conc = time.time() - start
    findings_conc = sum(1 for r in results_conc if r.detected and r.confidence >= 0.7)
    logger.success(f"Concurrent: {time_conc:.1f}s | {findings_conc} findings")

    # Results
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Sequential:  {time_seq:.1f}s")
    logger.info(f"Concurrent:  {time_conc:.1f}s")
    logger.info(f"Speedup:     {time_seq / time_conc:.1f}x")
    logger.info(f"Time saved:  {time_seq - time_conc:.1f}s")
    logger.info("")
    logger.info("Extrapolated for all 44 patterns:")
    logger.info(f"  Sequential: ~{time_seq * 44 / len(patterns):.0f}s")
    logger.info(f"  Concurrent: ~{time_conc * 44 / len(patterns):.0f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
