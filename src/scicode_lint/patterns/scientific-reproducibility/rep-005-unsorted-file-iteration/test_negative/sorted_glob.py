import json
from pathlib import Path


class ManifestDataLoader:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def get_files(self) -> list[Path]:
        base_dir = self.manifest_path.parent
        return [base_dir / name for name in self.manifest["files"]]


def load_numbered_files(directory: Path, prefix: str = "data") -> list[Path]:
    files = list(directory.glob(f"{prefix}_*.npy"))
    return sorted(files, key=lambda p: int(p.stem.split("_")[-1]))


def load_timestamped_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.csv"))
    return sorted(files, key=lambda p: p.stat().st_mtime)
