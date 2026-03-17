"""Genomics pipeline sorting discrete expression levels without stable sort."""

import numpy as np


def rank_predictions(gene_expressions: np.ndarray, gene_ids: np.ndarray) -> list[str]:
    """Rank genes by discretized expression level - many ties after rounding."""
    levels = np.round(gene_expressions).astype(int)
    order = np.argsort(levels)[::-1]
    return [gene_ids[i] for i in order]


def assign_percentile_ranks(patient_ages: np.ndarray) -> np.ndarray:
    """Assign percentile ranks to patients - integer ages have many ties."""
    return np.argsort(np.argsort(patient_ages))


def sort_variants_by_frequency(variant_counts: np.ndarray) -> np.ndarray:
    """Sort genetic variants by occurrence count - integer counts have ties."""
    return np.sort(variant_counts)[::-1]


def prioritize_candidates(priority_scores: np.ndarray, n_select: int) -> np.ndarray:
    """Select top-N drug candidates by integer priority score."""
    ranked = np.argsort(-priority_scores)
    return ranked[:n_select]
