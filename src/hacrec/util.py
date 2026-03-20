"""Shared filesystem utility helpers."""

from pathlib import Path


def project_root() -> Path:
    """Return the project root from this source file location."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(root: Path, subdir: str) -> Path:
    """Create and return a subdirectory under *root*."""
    dir_path = root / subdir
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
