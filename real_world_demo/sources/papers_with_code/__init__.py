"""Papers With Code data source.

Scripts for downloading and preparing scientific ML repositories from PapersWithCode.

Usage:
    # Full pipeline
    python -m real_world_demo.sources.papers_with_code --run --papers 50

    # Run analysis
    PYTHONPATH=. python real_world_demo/run_analysis.py \
        --manifest real_world_demo/collected_code/manifest.csv \
        --base-dir real_world_demo/collected_code \
        --max-concurrent 50
"""
