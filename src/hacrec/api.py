"""Hackneyed Recommender — API server.

A small FastAPI backend that serves the webapp and exposes endpoints:

  GET  /api/movies          → list of all movies with id, title, genres
  POST /api/recommend       → top-10 recommendations for a new user's ratings
  GET  /api/recommenders    → all registered algorithms with their hyperparameter schemas
  POST /api/fit             → fit a model with given hyperparameters and save a checkpoint
  GET  /api/models          → list all fitted model checkpoints

Run with:
    uv run hacrecapi
"""

import csv
import pathlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from .load import load_mapping, load_user_item_matrix
from .recommender_registry import registry
from .util import project_root

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT = project_root()
OUT_DIR = ROOT / "out"
DATA_DIR = ROOT / "data" / "ml-latest-small"
WEBAPP_DIR = ROOT / "webapp"

# Hints that enrich each hyperparameter's schema for the playground UI.
_PARAM_HINTS: dict[str, dict] = {
    "k":            {"min": 1,      "max": 200,  "step": 1},
    "n_factors":    {"min": 1,      "max": 300,  "step": 1},
    "n_iterations": {"min": 1,      "max": 200,  "step": 1},
    "n_epochs":     {"min": 1,      "max": 200,  "step": 1},
    "seed":         {"min": 0,      "max": 9999, "step": 1},
    "lambda_":      {"min": 0.0,    "max": 10.0, "step": 0.001},
    "reg":          {"min": 0.0,    "max": 100.0,"step": 0.1},
    "lr":           {"min": 0.0001, "max": 1.0,  "step": 0.001},
    "alpha":        {"min": 0.0,    "max": 100.0,"step": 0.1},
}

# ------------------------------------------------------------------
# Startup data loading
# ------------------------------------------------------------------

def _load_movies() -> list[dict]:
    """Load movies.csv into a list of {id, title, genres}."""
    df = pd.read_csv(DATA_DIR / "movies.csv")
    movies = []
    for row in df.itertuples(index=False):
        genres = [g for g in str(row.genres).split("|") if g and g != "(no genres listed)"]
        movies.append({"id": int(row.movieId), "title": str(row.title), "genres": genres})
    return movies


def _load_item_mapping() -> dict[int, int]:
    """Return original movieId → mapped column index."""
    raw = load_mapping(str(OUT_DIR / "item_mapping.csv"))
    return {int(k): int(v) for k, v in raw.items()}


# Pre-load everything once at import time so the server starts fast.
MOVIES: list[dict] = _load_movies()
MOVIE_LOOKUP: dict[int, dict] = {m["id"]: m for m in MOVIES}
ITEM_MAPPING: dict[int, int] = _load_item_mapping()          # originalId → col
ITEM_MAPPING_REV: dict[int, int] = {v: k for k, v in ITEM_MAPPING.items()}  # col → originalId
URM: sp.csr_matrix = load_user_item_matrix(OUT_DIR)

ALGORITHM_LABELS: dict[str, str] = {
    name: str(registry.build(name))
    for name in registry.names
}

# ------------------------------------------------------------------
# models.csv helpers
# ------------------------------------------------------------------

def _read_models_csv(checkpoint_dir: pathlib.Path) -> list[dict]:
    csv_path = checkpoint_dir / "models.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _coerce_params(name: str, raw: dict) -> dict:
    """Cast param values to the types declared in the registry defaults."""
    _, defaults = registry._entries[name]
    out: dict = {}
    for k, v in raw.items():
        if k in defaults:
            d = defaults[k]
            if isinstance(d, bool):
                out[k] = bool(v)
            elif isinstance(d, int):
                out[k] = int(v)
            elif isinstance(d, float):
                out[k] = float(v)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class RecommendRequest(BaseModel):
    algorithm: str                  # "item-based-cf" | "als"
    params: dict = {}               # hyperparameter overrides (from playground manifest)
    ratings: dict[str, int]         # { "<movieId>": 1-5 }


class FitRequest(BaseModel):
    name: str
    params: dict = {}
    force_refit: bool = False


class RecommendedMovie(BaseModel):
    rank: int
    movieId: int
    title: str
    genres: list[str]
    score: float


class RecommendResponse(BaseModel):
    algorithm: str
    recommendations: list[RecommendedMovie]

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------

app = FastAPI(title="Hackneyed Recommender API")


@app.get("/api/movies")
def get_movies() -> list[dict]:
    """Return every movie in the dataset."""
    return MOVIES


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Generate top-10 recommendations for a brand-new user.

    1.  Map the user's ratings (keyed by original movieId) into the URM's
        column-index space.
    2.  Append a new row to the URM representing this user.
    3.  Fit the chosen model on the augmented URM.
    4.  Call model.recommend(new_user_id, n=10).
    5.  Map internal item indices back to movie metadata and return.
    """
    if req.algorithm not in registry:
        raise HTTPException(400, f"Unknown algorithm '{req.algorithm}'. Choose from: {list(ALGORITHM_LABELS)}")

    # --- Translate user ratings into URM column indices ----------------
    new_row_data: list[float] = []
    new_row_cols: list[int] = []

    for movie_id_str, rating in req.ratings.items():
        movie_id = int(movie_id_str)
        col = ITEM_MAPPING.get(movie_id)
        if col is None:
            # Movie exists in the catalogue but wasn't in the training set
            # (e.g. no one rated it enough). Skip silently.
            continue
        new_row_data.append(float(rating))
        new_row_cols.append(col)

    if not new_row_data:
        raise HTTPException(
            422,
            "None of the rated movies appear in the training data. "
            "Try rating some more popular titles.",
        )

    # --- Build augmented URM (original + one new user row) -------------
    n_users, n_items = URM.shape
    new_user_id = n_users  # index of the appended row

    new_row = sp.csr_matrix(
        (new_row_data, ([0] * len(new_row_data), new_row_cols)),
        shape=(1, n_items),
    )
    augmented_urm = sp.vstack([URM, new_row], format="csr")

    # --- Fit model on the augmented URM --------------------------------
    coerced = _coerce_params(req.algorithm, req.params)
    model = registry.build(req.algorithm, **coerced)
    model.fit(augmented_urm)

    # --- Get top-10 recommendations ------------------------------------
    raw_recs = model.recommend(new_user_id, n=10)

    results: list[RecommendedMovie] = []
    for rank, (col_idx, score) in enumerate(raw_recs, start=1):
        original_id = ITEM_MAPPING_REV.get(col_idx)
        movie = MOVIE_LOOKUP.get(original_id, {}) if original_id is not None else {}
        results.append(RecommendedMovie(
            rank=rank,
            movieId=original_id or col_idx,
            title=movie.get("title", f"Movie #{col_idx}"),
            genres=movie.get("genres", []),
            score=round(float(score), 4),
        ))

    return RecommendResponse(algorithm=req.algorithm, recommendations=results)


@app.get("/api/recommenders")
def get_recommenders():
    """Return every registered algorithm with its hyperparameter schema."""
    result = []
    for name, (cls, defaults) in registry._entries.items():
        param_defs = []
        for k, v in defaults.items():
            if isinstance(v, bool):
                ptype, hints = "bool", {}
            elif isinstance(v, int):
                ptype, hints = "int", _PARAM_HINTS.get(k, {})
            else:
                ptype, hints = "float", _PARAM_HINTS.get(k, {})
            param_defs.append({"name": k, "type": ptype, "default": v, **hints})
        result.append({
            "name": name,
            "label": ALGORITHM_LABELS.get(name, name),
            "params": param_defs,
        })
    return {"recommenders": result}


@app.post("/api/fit")
def fit_model(req: FitRequest):
    """Fit a recommender on the base URM and persist a checkpoint + models.csv entry."""
    from .fit import fit_recommender, default_checkpoint_dir

    if req.name not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown algorithm '{req.name}'")

    coerced = _coerce_params(req.name, req.params)
    _, defaults = registry._entries[req.name]
    effective = {**defaults, **coerced}

    checkpoint_dir = default_checkpoint_dir()
    fit_recommender(req.name, URM, checkpoint_dir, force_refit=req.force_refit, **coerced)

    # Find the last matching entry in models.csv (covers both new fit and loaded cache)
    rows = _read_models_csv(checkpoint_dir)
    matching = [r for r in rows if r["name"] == req.name and eval(r["params"]) == effective]  # noqa: S307
    row = matching[-1] if matching else None

    folder_name = f"{req.name}_{row['datetime']}" if row else req.name
    return {
        "id": folder_name,
        "name": req.name,
        "label": ALGORITHM_LABELS.get(req.name, req.name),
        "datetime": row["datetime"] if row else "",
        "params": effective,
        "folder": folder_name,
    }


@app.get("/api/models")
def list_models():
    """Return all fitted model entries from models.csv."""
    from .fit import default_checkpoint_dir

    checkpoint_dir = default_checkpoint_dir()
    rows = _read_models_csv(checkpoint_dir)
    models = []
    for row in rows:
        try:
            params = eval(row["params"])  # noqa: S307
        except Exception:
            params = {}
        folder_name = f"{row['name']}_{row['datetime']}"
        models.append({
            "id": folder_name,
            "name": row["name"],
            "label": ALGORITHM_LABELS.get(row["name"], row["name"]),
            "datetime": row["datetime"],
            "params": params,
            "folder": folder_name,
        })
    return {"models": models}


# ------------------------------------------------------------------
# Serve the frontend (static files + SPA fallback)
# ------------------------------------------------------------------

# Serve static assets (style.css, app.js) at the root so relative paths in
# index.html work regardless of whether the user visits / or /index.html.
app.mount("/webapp", StaticFiles(directory=str(WEBAPP_DIR)), name="webapp-prefix")
app.mount("/", StaticFiles(directory=str(WEBAPP_DIR), html=True), name="webapp")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    import uvicorn
    print("Starting Hackneyed Recommender API …")
    print("Open http://localhost:8000 in your browser.\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
