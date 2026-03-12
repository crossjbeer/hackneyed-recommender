# Hackneyed Recommender

A simple, __hackneyed__ recommender system for the MovieLens Dataset.
Using Collaborative Filtering and Matrix Factorization. 
Demonstrating comparative recommender performance and various use cases. 
An exploratory project. 

## How it works
- `prepare.py`   -- Extracts the MovieLens dataset
- `transform.py` -- Transforms raw .csv datasets into pandas dataframes. Builds our User-Item Rating Matrix (URM)
- `Recommendation Engines`
  - `collaborativefiltering.py` -- Runs Collaborative Filtering over our URM
  - `factorization.py`          -- Runs Matrix Factorization using Alternating Least Squares (ALS) over our URM 
- `eval.py` -- A simple endpoint to run and compare various recommendation engines

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install UV (if you don't have it)
# Windows: winget install --id=astral-sh.uv
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone <repo-url>
cd basic-recommender
uv sync # Creates .venv automatically 
```

## Usage

```bash
# Run the evaluation script
uv run hacreceval

# Run any python script
uv run python src/hacrec/prepare.py
```