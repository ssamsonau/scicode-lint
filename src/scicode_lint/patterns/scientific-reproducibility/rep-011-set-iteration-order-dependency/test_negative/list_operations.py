"""Operations using ordered collections - no set iteration."""

import numpy as np
from collections import Counter


def rank_by_frequency(words: list[str]) -> list[tuple[str, int]]:
    """Rank words by frequency using Counter (dict-based, insertion-ordered)."""
    counts = Counter(words)
    return counts.most_common()


def deduplicate_preserving_order(items: list[str]) -> list[str]:
    """Remove duplicates while preserving original list order."""
    seen = {}
    result = []
    for item in items:
        if item not in seen:
            seen[item] = True
            result.append(item)
    return result


def aggregate_metrics(metric_names: list[str], values: np.ndarray) -> dict[str, float]:
    """Aggregate metrics using zip with two ordered lists."""
    return {name: float(val) for name, val in zip(metric_names, values.mean(axis=0))}
