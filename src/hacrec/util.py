"""Shared filesystem utility helpers."""

from pathlib import Path

import pandas as pd


def project_root() -> Path:
    """Return the project root from this source file location."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(root: Path, subdir: str) -> Path:
    """Create and return a subdirectory under *root*."""
    dir_path = root / subdir
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_mapping(mapping: dict, filename: str) -> None:
    """Save a mapping dictionary to a CSV file."""
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['original_id', 'mapped_id'])
    mapping_df.to_csv(filename, index=False)


def load_mapping(filename: str) -> dict:
    """Load a mapping dictionary from a CSV file."""
    mapping_df = pd.read_csv(filename)
    return dict(zip(mapping_df['original_id'], mapping_df['mapped_id']))
