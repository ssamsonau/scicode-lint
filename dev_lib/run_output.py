"""Shared run output infrastructure for disk-streaming results.

Provides RunOutput (timestamped output directories) and an async write worker
for serializing disk writes across concurrent tasks.

Used by: semantic_validate.py, diversity_check.py, integration_eval.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiofiles


@dataclass
class RunOutput:
    """Output paths for a timestamped run directory.

    Standard structure:
        reports/<timestamp>_<scope>/
        ├── summary.md          # Final summary
        ├── progress.log        # One line per item as it completes
        └── items/              # Per-item log files
            ├── item_1.log
            └── ...

    The items_dir name is configurable (e.g., "patterns/", "scenarios/").
    """

    run_dir: Path
    summary: Path
    log: Path
    items_dir: Path

    @classmethod
    def create(
        cls,
        reports_dir: Path,
        scope: str,
        items_dirname: str = "items",
    ) -> RunOutput:
        """Create a new run output directory structure.

        Args:
            reports_dir: Parent directory for run outputs (e.g., reports/).
            scope: What's being run (e.g., "all", "ai-training", "generate_10").
            items_dirname: Name for the per-item subdirectory (e.g., "patterns", "scenarios").

        Returns:
            RunOutput with all paths created.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = reports_dir / f"{timestamp}_{scope}"
        items_dir = run_dir / items_dirname

        run_dir.mkdir(parents=True, exist_ok=True)
        items_dir.mkdir(exist_ok=True)

        return cls(
            run_dir=run_dir,
            summary=run_dir / "summary.md",
            log=run_dir / "progress.log",
            items_dir=items_dir,
        )

    def item_file(self, name: str, ext: str = ".log") -> Path:
        """Get path for an individual item log file.

        Args:
            name: Item identifier (e.g., pattern ID, scenario name).
            ext: File extension (default: ".log").

        Returns:
            Path to the item file.
        """
        return self.items_dir / f"{name}{ext}"

    def init_log(self) -> None:
        """Initialize the progress log file (create empty)."""
        self.log.write_text("")


async def write_worker(
    queue: asyncio.Queue[tuple[Path, str] | None],
) -> None:
    """Worker that writes files sequentially from a queue.

    Serializes disk writes to avoid I/O contention. Multiple producers can
    enqueue writes concurrently; this worker processes them one at a time.

    Send None to signal the worker to stop.

    Args:
        queue: Queue of (path, content) tuples, or None to stop.
    """
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        path, content = item
        try:
            async with aiofiles.open(path, "w") as f:
                await f.write(content)
        except OSError as e:
            print(f"Warning: failed to write {path}: {e}", flush=True)
        queue.task_done()
