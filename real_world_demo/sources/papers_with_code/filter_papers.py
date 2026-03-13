"""Filter scientific ML papers from PapersWithCode archive.

Loads HuggingFace datasets, filters by scientific domains, and extracts repo URLs.

Usage:
    python filter_papers.py [--limit 100] [--domains biology,medical]
"""

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from real_world_demo.config import (
    DATA_DIR,
    EXCLUDE_TASK_KEYWORDS,
    EXCLUDE_VENUES,
    HF_LINKS_DATASET,
    HF_PAPERS_DATASET,
    SCIENTIFIC_DOMAINS,
)


def load_datasets(
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load both HuggingFace datasets.

    HuggingFace datasets library automatically caches downloads, so subsequent
    runs use cached data without re-downloading.

    Args:
        cache_dir: Optional custom cache directory for datasets.

    Returns:
        Tuple of (papers, links) as lists of dicts.

    Raises:
        ImportError: If datasets library not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library required. Install with: pip install -e '.[real-world-demo]'"
        ) from e

    # HuggingFace datasets library caches by default to ~/.cache/huggingface/datasets
    # Subsequent runs will use cached data without re-downloading
    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}

    logger.info(f"Loading {HF_PAPERS_DATASET}... (uses HF cache if available)")
    papers_ds = load_dataset(HF_PAPERS_DATASET, split="train", **cache_kwargs)
    papers = list(papers_ds)
    logger.info(f"Loaded {len(papers):,} papers")

    logger.info(f"Loading {HF_LINKS_DATASET}... (uses HF cache if available)")
    links_ds = load_dataset(HF_LINKS_DATASET, split="train", **cache_kwargs)
    links = list(links_ds)
    logger.info(f"Loaded {len(links):,} paper-code links")

    return papers, links


def matches_domain(tasks: list[str] | None, domains: list[str] | None = None) -> str | None:
    """Check if tasks match any scientific domain.

    Args:
        tasks: List of task names from PWC paper.
        domains: Optional list of domains to filter (None = all).

    Returns:
        Domain name if matched, None otherwise.
    """
    if not tasks:
        return None

    # Build domain keywords to check
    if domains:
        domain_keywords = {d: SCIENTIFIC_DOMAINS[d] for d in domains if d in SCIENTIFIC_DOMAINS}
    else:
        domain_keywords = SCIENTIFIC_DOMAINS

    # Convert tasks to lowercase for matching
    tasks_lower = [t.lower() for t in tasks]
    tasks_text = " ".join(tasks_lower)

    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword.lower() in tasks_text:
                return domain

    return None


def should_exclude(tasks: list[str] | None) -> bool:
    """Check if tasks indicate non-scientific ML work.

    Args:
        tasks: List of task names from PWC paper.

    Returns:
        True if paper should be excluded.
    """
    if not tasks:
        return False

    tasks_text = " ".join(t.lower() for t in tasks)

    for keyword in EXCLUDE_TASK_KEYWORDS:
        if keyword.lower() in tasks_text:
            return True

    return False


def is_ml_venue(paper: dict[str, Any]) -> bool:
    """Check if paper is from an ML/AI venue that should be excluded.

    We want science journals/venues, not ML conferences. This excludes papers
    from venues like NeurIPS, ICML, ICLR, etc.

    Args:
        paper: Paper record with conference/proceeding fields.

    Returns:
        True if paper is from an ML venue (should be excluded).
    """
    # Check conference field
    conference = paper.get("conference", "") or ""
    proceeding = paper.get("proceeding", "") or ""

    # Combine and lowercase for matching
    venue_text = f"{conference} {proceeding}".lower()

    if not venue_text.strip():
        # No venue info - keep for now (will be filtered by abstract later)
        return False

    for venue in EXCLUDE_VENUES:
        if venue.lower() in venue_text:
            return True

    return False


def _balanced_sample(
    papers_by_domain: dict[str, list[dict[str, Any]]],
    limit: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample papers evenly across domains.

    Args:
        papers_by_domain: Dict mapping domain name to list of papers.
        limit: Total number of papers to return.
        seed: Random seed for reproducibility.

    Returns:
        List of papers sampled evenly across domains.
    """
    import random

    random.seed(seed)
    num_domains = len(papers_by_domain)
    per_domain = limit // num_domains
    remainder = limit % num_domains

    result: list[dict[str, Any]] = []
    domains = list(papers_by_domain.keys())

    # Shuffle domains so remainder distribution is random
    random.shuffle(domains)

    for i, domain in enumerate(domains):
        domain_papers = papers_by_domain[domain]
        # Give one extra to first 'remainder' domains
        take = per_domain + (1 if i < remainder else 0)
        # Shuffle and take
        shuffled = domain_papers.copy()
        random.shuffle(shuffled)
        result.extend(shuffled[:take])

    # Shuffle final result so domains are interleaved
    random.shuffle(result)
    return result


def filter_papers(
    papers: list[dict[str, Any]],
    links: list[dict[str, Any]],
    domains: list[str] | None = None,
    limit: int | None = None,
    balanced: bool = True,
) -> list[dict[str, Any]]:
    """Filter papers to scientific domains with embedded repo URLs.

    Filters papers by:
    1. Scientific domain keywords (biology, chemistry, medical, etc.)
    2. Excludes ML venues (NeurIPS, ICML, ICLR, etc.)
    3. Requires GitHub code links

    Each paper record includes repo_urls embedded directly.

    Args:
        papers: List of paper records from HuggingFace.
        links: List of paper-code link records from HuggingFace.
        domains: Optional list of domains to include (None = all).
        limit: Optional limit on number of papers to return.
        balanced: If True and limit is set, sample evenly across domains.

    Returns:
        List of filtered paper records with repo_urls embedded.
    """
    # Build paper lookup by paper_url (the join key)
    logger.info("Building paper index...")
    paper_by_url: dict[str, dict[str, Any]] = {}
    for paper in papers:
        paper_url = paper.get("paper_url")
        if paper_url:
            paper_by_url[paper_url] = paper

    logger.info(f"Indexed {len(paper_by_url):,} papers by URL")

    # Build link lookup by paper_url
    logger.info("Building link index...")
    links_by_paper: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        paper_url = link.get("paper_url")
        if paper_url:
            if paper_url not in links_by_paper:
                links_by_paper[paper_url] = []
            links_by_paper[paper_url].append(link)

    logger.info(f"Indexed {len(links_by_paper):,} papers with code links")

    # Filter papers by domain - only include papers WITH code links
    logger.info("Filtering by scientific domain (only papers with code links)...")
    logger.info("Excluding ML venues (NeurIPS, ICML, ICLR, etc.)...")
    papers_by_domain: dict[str, list[dict[str, Any]]] = {}
    ml_venue_excluded = 0

    for paper in papers:
        paper_url = paper.get("paper_url")

        # Skip papers without code links (critical for balanced sampling to work)
        if not paper_url or paper_url not in links_by_paper:
            continue

        tasks = paper.get("tasks", [])

        # Skip if matches exclusion criteria
        if should_exclude(tasks):
            continue

        # Skip papers from ML venues (we want science venues)
        if is_ml_venue(paper):
            ml_venue_excluded += 1
            continue

        # Check if matches a scientific domain
        domain = matches_domain(tasks, domains)
        if domain:
            paper["matched_domain"] = domain
            # Embed repo URLs from links into paper record
            repo_urls = [
                normalize_github_url(link.get("repo_url", ""))
                for link in links_by_paper.get(paper_url, [])
                if link.get("repo_url")
            ]
            paper["repo_urls"] = repo_urls
            if domain not in papers_by_domain:
                papers_by_domain[domain] = []
            papers_by_domain[domain].append(paper)

    # Log what we found
    total_found = sum(len(p) for p in papers_by_domain.values())
    logger.info(f"Excluded {ml_venue_excluded:,} papers from ML venues")
    logger.info(f"Found {total_found:,} papers in scientific domains")
    for domain, domain_papers in sorted(papers_by_domain.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {domain}: {len(domain_papers):,}")

    # Apply limit with balanced sampling if requested
    if limit and balanced and len(papers_by_domain) > 1:
        filtered_papers = _balanced_sample(papers_by_domain, limit)
        n_papers = len(filtered_papers)
        n_domains = len(papers_by_domain)
        logger.info(f"Balanced sampling: {n_papers:,} papers across {n_domains} domains")
    elif limit:
        # Simple limit without balancing
        filtered_papers = []
        for domain_papers in papers_by_domain.values():
            filtered_papers.extend(domain_papers)
            if len(filtered_papers) >= limit:
                break
        filtered_papers = filtered_papers[:limit]
    else:
        # No limit - take all
        filtered_papers = []
        for domain_papers in papers_by_domain.values():
            filtered_papers.extend(domain_papers)

    # Log final distribution
    domain_counts: dict[str, int] = {}
    total_repos = 0
    for paper in filtered_papers:
        d = paper.get("matched_domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
        total_repos += len(paper.get("repo_urls", []))
    logger.info(f"Final selection: {len(filtered_papers):,} papers with {total_repos:,} repo URLs")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {domain}: {count:,}")

    return filtered_papers


def normalize_github_url(url: str) -> str:
    """Normalize GitHub URL to canonical form.

    Args:
        url: GitHub repo URL in various formats.

    Returns:
        Normalized URL (https://github.com/owner/repo).
    """
    url = url.rstrip("/")

    # Remove .git suffix
    if url.endswith(".git"):
        url = url[:-4]

    # Convert git:// to https://
    if url.startswith("git://github.com"):
        url = url.replace("git://", "https://")

    # Handle github.com:owner/repo format
    if url.startswith("git@github.com:"):
        url = url.replace("git@github.com:", "https://github.com/")

    return url


def _make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format.

    Handles Timestamp and other non-serializable types from HuggingFace datasets.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable version.
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif hasattr(obj, "isoformat"):  # datetime, Timestamp, etc.
        return obj.isoformat()
    elif hasattr(obj, "__str__") and not isinstance(obj, (str, int, float, bool, type(None))):
        # For other non-standard types, convert to string
        return str(obj)
    return obj


def save_results(
    filtered_papers: list[dict[str, Any]],
    output_dir: Path = DATA_DIR,
) -> None:
    """Save filtered papers to JSON file.

    Each paper record includes repo_urls embedded directly.

    Args:
        filtered_papers: List of filtered paper records (with repo_urls).
        output_dir: Directory to save output files.
    """
    output_dir.mkdir(exist_ok=True)

    # Convert to JSON-serializable format (handles Timestamps from HuggingFace)
    papers_serializable = _make_json_serializable(filtered_papers)

    papers_file = output_dir / "filtered_papers.json"
    with open(papers_file, "w") as f:
        json.dump(papers_serializable, f, indent=2)
    logger.info(f"Saved {len(filtered_papers):,} papers to {papers_file}")


def main() -> None:
    """Main entry point for paper filtering."""
    parser = argparse.ArgumentParser(description="Filter scientific ML papers from PWC archive")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of papers to process",
    )
    parser.add_argument(
        "--domains",
        type=str,
        help="Comma-separated list of domains to include (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-filtering even if output files exist",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Custom cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="Disable balanced sampling across domains (default: balanced)",
    )
    args = parser.parse_args()

    # Check if output already exists
    papers_file = args.output_dir / "filtered_papers.json"
    if papers_file.exists() and not args.force:
        logger.info(f"Output already exists: {papers_file}")
        logger.info("Use --force to re-filter, or delete the file manually")
        # Load and print summary of existing data
        with open(papers_file) as f:
            existing_papers = json.load(f)
        logger.info(f"Existing papers: {len(existing_papers):,}")
        return

    domains = args.domains.split(",") if args.domains else None

    # Load datasets (uses HF cache automatically)
    papers, links = load_datasets(cache_dir=args.cache_dir)

    # Filter papers (balanced sampling by default when limit is set)
    balanced = not args.no_balanced
    filtered_papers = filter_papers(
        papers, links, domains=domains, limit=args.limit, balanced=balanced
    )

    # Save results
    save_results(filtered_papers, args.output_dir)

    # Summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"  Total papers filtered: {len(filtered_papers):,}")
    total_repos = sum(len(p.get("repo_urls", [])) for p in filtered_papers)
    logger.info(f"  Total repo URLs: {total_repos:,}")
    domain_dist: dict[str, int] = {}
    for paper in filtered_papers:
        d = paper.get("matched_domain", "unknown")
        domain_dist[d] = domain_dist.get(d, 0) + 1
    logger.info("  Papers by domain:")
    for domain, count in sorted(domain_dist.items(), key=lambda x: -x[1]):
        logger.info(f"    {domain}: {count:,} ({100 * count / len(filtered_papers):.1f}%)")


if __name__ == "__main__":
    main()
