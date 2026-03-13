from pathlib import Path

import pandas as pd
import pytest

from hacrec.util import ensure_dir, load_mapping, save_mapping


def test_ensure_dir_creates_nested_directory_in_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    created = ensure_dir(Path.cwd(), "alpha/beta")

    assert created == tmp_path / "alpha" / "beta"
    assert created.exists()
    assert created.is_dir()


def test_ensure_dir_is_idempotent_for_existing_directory(tmp_path):
    root = tmp_path

    first = ensure_dir(root, "cache")
    second = ensure_dir(root, "cache")

    assert first == second
    assert second.exists()
    assert second.is_dir()


def test_save_mapping_creates_csv_with_correct_columns(tmp_path):
    mapping = {1: 0, 5: 1, 9: 2}
    filepath = str(tmp_path / "mapping.csv")

    save_mapping(mapping, filepath)

    df = pd.read_csv(filepath)
    assert list(df.columns) == ["original_id", "mapped_id"]


def test_save_mapping_writes_all_entries(tmp_path):
    mapping = {10: 0, 20: 1, 30: 2}
    filepath = str(tmp_path / "mapping.csv")

    save_mapping(mapping, filepath)

    df = pd.read_csv(filepath)
    assert len(df) == len(mapping)
    assert set(df["original_id"]) == set(mapping.keys())
    assert set(df["mapped_id"]) == set(mapping.values())


def test_load_mapping_returns_correct_dict(tmp_path):
    filepath = str(tmp_path / "mapping.csv")
    df = pd.DataFrame({"original_id": [1, 2, 3], "mapped_id": [10, 20, 30]})
    df.to_csv(filepath, index=False)

    result = load_mapping(filepath)

    assert result == {1: 10, 2: 20, 3: 30}


def test_save_and_load_mapping_roundtrip(tmp_path):
    original = {42: 0, 7: 1, 99: 2}
    filepath = str(tmp_path / "mapping.csv")

    save_mapping(original, filepath)
    restored = load_mapping(filepath)

    assert restored == original
