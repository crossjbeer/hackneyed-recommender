"""Centralized data loaders and savers for persisted artifacts."""

from pathlib import Path

import pandas as pd
import scipy.sparse as sp


def save_mapping(mapping: dict, filename: str | Path) -> None:
    """Save a mapping dictionary to a CSV file."""
    mapping_df = pd.DataFrame(list(mapping.items()), columns=["original_id", "mapped_id"])
    mapping_df.to_csv(filename, index=False)


def load_mapping(filename: str | Path) -> dict:
    """Load a mapping dictionary from a CSV file."""
    mapping_df = pd.read_csv(filename)
    return dict(zip(mapping_df["original_id"], mapping_df["mapped_id"]))


def load_ratings_splits(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test rating splits from the output directory."""
    out = Path(out_dir)
    train_df = pd.read_csv(out / "train_ratings.csv")
    val_df = pd.read_csv(out / "val_ratings.csv")
    test_df = pd.read_csv(out / "test_ratings.csv")
    return train_df, val_df, test_df


def load_user_item_matrix(out_dir: str | Path) -> sp.csr_matrix:
    """Load the saved user-item interaction matrix from the output directory."""
    out = Path(out_dir)
    return sp.load_npz(out / "user_item_matrix.npz")
