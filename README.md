# Hackneyed Recommender

How many times have you wanted to compare the relative performance of several recommendation strategies on the MovieLens Dataset? 
Never?  

Well now you can! 

Fit recommenders yourself and evaluate exciting metrics like NDCG@10, RMSE, and more! 
See how `Prediction` models compare against `Ranking` models. 
Try out the recommenders yourself in our interactive playground! 

## How it works
- `prepare.py`   -- Extracts the MovieLens dataset
- `transform.py` -- Transforms raw .csv datasets into pandas dataframes. Builds our User-Item Rating Matrix (URM) 
- `fit.py`       -- Fit and checkpoint all registry `Recommenders` with default parameters
- `eval.py`      -- Evaluate fitted `Recommenders` and persist comparison metrics
- `visualize.py` -- Produce an information dashboard to visualize performance
- `api.py`       -- FastAPI to interact with our Playground Frontend

### Recommendation Engines
There are two classes of `Recommenders`, one optimizing for `Prediction`, the other optimizing for `Ranking`. 

The Abstract Base Class (ABC) is represented at `recommender.py`. Specific instances are listed below: 

- Ranking / Prediction: 
  - `itembasedcf.py`                  -- Item-Based CF using Cosine Similarity 
  - `biaseditembasedcf.py`            -- Biased Item-Based CF using Baseline-Corrected Residuals 
  - `alsfactorization.py`             -- Matrix Factorization using Alternating Least Squares (ALS)
  - `biasedalsfactorization.py`       -- Biased Matrix Factorization using Baseline-Corrected Residuals 
- Ranking: 
  - `implicitalsfactorization.py`     -- Matrix Factorization using Implicit ALS 
  - `bprfactorization.py`             -- Matrix Factorization using Bayesian Personalized Ranking (BPR)
  - `adjustedbprfactorization.py`     -- BPR Factorization with adjustements to reduce Popularity Bias

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
# Fit and checkpoint all recommenders with registry defaults
uv run hacrecfit

# Run the evaluation script
uv run hacreceval

# Run any python script
uv run python src/hacrec/prepare.py
```