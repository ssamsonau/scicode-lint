"""Leakage paper data source (Yang et al. ASE'22).

Scripts for downloading, preparing, and evaluating against the leakage paper dataset.

Usage:
    # Download and prepare data
    python -m real_world_demo.sources.leakage_paper --run

    # Run analysis
    PYTHONPATH=. python real_world_demo/run_analysis.py \
        --manifest real_world_demo/data/leakage_paper/manifest.csv \
        --base-dir real_world_demo/collected_code/leakage_paper \
        --patterns ml-009,ml-010,pt-001 \
        --max-concurrent 50

    # Compare with ground truth
    python -m real_world_demo.sources.leakage_paper.compare_ground_truth
    python -m real_world_demo.sources.leakage_paper.compare_ground_truth --detailed
"""
