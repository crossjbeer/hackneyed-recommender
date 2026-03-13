from pathlib import Path

from hacrec.util import ensure_dir


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
