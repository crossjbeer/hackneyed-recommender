"""Hackneyed Recommender — API server.

A small FastAPI backend that serves the webapp and exposes two endpoints:

  GET  /api/movies     → list of all movies with id, title, genres
  POST /api/recommend  → top-10 recommendations for a new user's ratings

Run with:
    uv run hacrecapi
"""

import pathlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from .recommender_registry import registry
from .util import load_mapping, project_root

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT = project_root()
OUT_DIR = ROOT / "out"
DATA_DIR = ROOT / "data" / "ml-latest-small"
WEBAPP_DIR = ROOT / "webapp"

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
URM: sp.csr_matrix = sp.load_npz(str(OUT_DIR / "user_item_matrix.npz"))

ALGORITHM_LABELS: dict[str, str] = {
    name: str(registry.build(name))
    for name in registry.names
}

# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class RecommendRequest(BaseModel):
    algorithm: str                  # "item-based-cf" | "als"
    ratings: dict[str, int]         # { "<movieId>": 1-5 }


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
    model = registry.build(req.algorithm)
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
