"""Acquire the MovieLens dataset and save it under a local data directory."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from .util import project_root, ensure_dir


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_movielens(data_dir: Path) -> Path:
	"""Download the MovieLens zip file into the data directory."""
	zip_path = data_dir / "ml-latest-small.zip"
	if not zip_path.exists():
		print(f"Downloading MovieLens dataset to {zip_path}...")
		urlretrieve(MOVIELENS_URL, zip_path)
	else:
		print(f"Zip already exists at {zip_path}; skipping download.")
	return zip_path


def extract_movielens(zip_path: Path, data_dir: Path) -> Path:
	"""Extract the MovieLens zip into the data directory."""
	extracted_dir = data_dir / "ml-latest-small"
	if not extracted_dir.exists():
		print(f"Extracting dataset into {data_dir}...")
		with ZipFile(zip_path, "r") as archive:
			archive.extractall(data_dir)
	else:
		print(f"Dataset already extracted at {extracted_dir}; skipping extraction.")
	return extracted_dir


def main() -> None:
	root = project_root()
	data_dir = ensure_dir(root, "data")
	zip_path = download_movielens(data_dir)
	extracted_dir = extract_movielens(zip_path, data_dir)
	print(f"MovieLens dataset is ready at {extracted_dir}")


if __name__ == "__main__":
	main()
