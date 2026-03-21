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
import re
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from .load import load_mapping, load_user_item_matrix
from .recommender_registry import registry
from .util import ensure_dir, project_root

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT = project_root()
OUT_DIR = ROOT / "out"
DATA_DIR = ROOT / "data" / "ml-latest-small"
WEBAPP_DIR = ROOT / "webapp"
VIZ_DIR = ensure_dir(OUT_DIR, "viz")

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
    model_id: str | None = None     # folder name of a pre-fitted checkpoint (optional)


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
# Checkpoint fold-in helper
# ------------------------------------------------------------------

def _fold_in_user(model: object, augmented_urm: sp.csr_matrix) -> None:
    """Mutate *model* in-place so model.recommend(n_users_original) works.

    The new user's ratings occupy the last row of *augmented_urm*.  Dispatches
    on the model's internal structure so each algorithm type gets a proper
    closed-form or approximate update:

    - Item-based CF / baselines   → update urm only (similarity pre-computed).
    - Plain ALS (has _solve)      → one closed-form user-factor step.
    - ImplicitALS (_update_factors) → confidence-weighted ALS fold-in.
    - BiasedALS (user_bias+item_factors) → bias + latent-factor fold-in.
    - BPR-style (user_factors only)  → initialise with mean user factors.
    """
    new_row_mat = augmented_urm[-1:]          # (1, n_items) csr_matrix
    model.urm = augmented_urm                 # type: ignore[attr-defined]

    user_factors = getattr(model, "user_factors", None)
    if user_factors is None:
        return  # item-based CF / baselines — urm update is sufficient

    rated_idx = new_row_mat.indices
    rated_vals = new_row_mat.data
    item_factors = getattr(model, "item_factors", None)

    # --- 1. BiasedALS: compute user_bias then latent factors ----------
    user_bias = getattr(model, "user_bias", None)
    if user_bias is not None and item_factors is not None:
        if len(rated_idx) > 0:
            new_bias = float(np.mean(
                rated_vals - model.global_mean - model.item_bias[rated_idx]  # type: ignore
            ))
        else:
            new_bias = 0.0
        model.user_bias = np.append(user_bias, new_bias)  # type: ignore
        if len(rated_idx) > 0:
            F = item_factors[rated_idx]
            residuals = (
                rated_vals
                - model.global_mean  # type: ignore
                - new_bias
                - model.item_bias[rated_idx]  # type: ignore
            )
            n_factors = item_factors.shape[1]
            reg = model.lambda_ * len(rated_idx) * np.eye(n_factors)  # type: ignore
            new_uv = np.linalg.solve(F.T @ F + reg, F.T @ residuals)
        else:
            new_uv = np.zeros(item_factors.shape[1])
        model.user_factors = np.vstack([user_factors, new_uv.reshape(1, -1)])  # type: ignore
        return

    # --- 2. Plain ALS: closed-form _solve ----------------------------
    if hasattr(model, "_solve") and item_factors is not None:
        new_factors = model._solve(new_row_mat, item_factors)  # (1, k)
        model.user_factors = np.vstack([user_factors, new_factors])  # type: ignore
        return

    # --- 3. ImplicitALS: confidence-weighted one-step fold-in --------
    if hasattr(model, "_update_factors") and item_factors is not None:
        k = item_factors.shape[1]
        VtV = item_factors.T @ item_factors
        alpha = getattr(model, "alpha", 40.0)
        reg = getattr(model, "lambda_", 0.1) * np.eye(k)
        if len(rated_idx) > 0:
            r_i = rated_vals
            c_i = 1.0 + alpha * r_i
            F_i = item_factors[rated_idx]
            A = VtV + F_i.T @ ((alpha * r_i)[:, None] * F_i) + reg
            new_uv = np.linalg.solve(A, F_i.T @ c_i)
        else:
            new_uv = np.zeros(k)
        model.user_factors = np.vstack([user_factors, new_uv.reshape(1, -1)])  # type: ignore
        return

    # --- 4. BPR-style: approximate with mean user factors ------------
    mean_uv = user_factors.mean(axis=0)
    model.user_factors = np.vstack([user_factors, mean_uv.reshape(1, -1)])  # type: ignore


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

    When *model_id* is supplied the named checkpoint is loaded from disk and
    the new user is folded in without re-training (fast).  Otherwise the model
    is built from *algorithm* + *params* and fitted fresh on the augmented URM
    (original behaviour, slower).
    """
    if not req.model_id and req.algorithm not in registry:
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

    # --- Load checkpoint or fit from scratch --------------------------------
    if req.model_id:
        # Validate to prevent path traversal
        if not re.fullmatch(r"[\w][\w\-]*_\d{6}_\d{6}", req.model_id):
            raise HTTPException(400, "Invalid model_id format.")
        from .fit import default_checkpoint_dir
        checkpoint_dir = default_checkpoint_dir()
        model_dir = (checkpoint_dir / req.model_id).resolve()
        if not str(model_dir).startswith(str(checkpoint_dir.resolve())):
            raise HTTPException(400, "Invalid model_id.")
        model_pkl = model_dir / "model.pkl"
        if not model_pkl.exists():
            raise HTTPException(404, f"Checkpoint '{req.model_id}' not found. Fit the model in the Playground first.")
        model = joblib.load(model_pkl)
        _fold_in_user(model, augmented_urm)
    else:
        # Legacy path: fit from scratch on the augmented URM
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
app.mount("/viz", StaticFiles(directory=str(VIZ_DIR), html=True), name="viz")
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
