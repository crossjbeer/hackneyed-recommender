"""Acquire the MovieLens dataset and save it under a local data directory."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from .util import project_root, ensure_dir


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_LARGE_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"


def download_movielens(data_dir: Path, large: bool = False, overwrite: bool = False) -> Path:
	"""Download the MovieLens zip file into the data directory."""
	url = MOVIELENS_LARGE_URL if large else MOVIELENS_SMALL_URL
	zip_name = "ml-latest.zip" if large else "ml-latest-small.zip"
	zip_path = data_dir / zip_name
	if not zip_path.exists() or overwrite:
		print(f"Downloading MovieLens dataset to {zip_path}...")
		urlretrieve(url, zip_path)
	else:
		print(f"Zip already exists at {zip_path}; skipping download.")
	return zip_path


def extract_movielens(zip_path: Path, data_dir: Path, large: bool = False) -> Path:
	"""Extract the MovieLens zip into the data directory."""
	extracted_dir = data_dir / ("ml-latest" if large else "ml-latest-small")
	if not extracted_dir.exists():
		print(f"Extracting dataset into {data_dir}...")
		with ZipFile(zip_path, "r") as archive:
			archive.extractall(data_dir)
	else:
		print(f"Dataset already extracted at {extracted_dir}; skipping extraction.")
	return extracted_dir


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Download and extract the MovieLens dataset.")
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Re-download the dataset even if it already exists locally.",
	)
	parser.add_argument(
		"--large",
		action="store_true",
		help="Download the full MovieLens dataset instead of the small version.",
	)
	return parser


def main() -> None:
	args = make_parser().parse_args()

	root = project_root()
	data_dir = ensure_dir(root, "data")
	zip_path = download_movielens(data_dir, large=args.large, overwrite=args.overwrite)
	extracted_dir = extract_movielens(zip_path, data_dir, large=args.large)
	print(f"MovieLens dataset is ready at {extracted_dir}")


if __name__ == "__main__":
	main()
