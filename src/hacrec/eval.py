"""Unified evaluation endpoint for all recommender strategies."""

import sys
import threading
import time
import pathlib as path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .recommender import Recommender


# ------------------------------------------------------------------
# Console spinner
# ------------------------------------------------------------------

class _Spinner:
    """Lightweight console spinner that writes to stderr."""

    _FRAMES = ("|", "/", "-", "\\")

    def __init__(self, message: str) -> None:
        self._message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._t_start: float = 0.0

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            elapsed = time.perf_counter() - self._t_start
            print(f"\r  [{frame}] {self._message} ({elapsed:.1f}s)", end="", flush=True, file=sys.stderr)
            i += 1
            self._stop.wait(0.1)

    def start(self) -> None:
        self._t_start = time.perf_counter()
        self._thread.start()

    def stop(self, elapsed: float) -> None:
        self._stop.set()
        self._thread.join()
        print(f"\r  [done] {self._message} — {elapsed:.2f}s", file=sys.stderr)


# ------------------------------------------------------------------
# Model-agnostic evaluation loop
# ------------------------------------------------------------------

def evaluate_predictions(model: Recommender, eval_df: pd.DataFrame) -> dict:
    """Compute RMSE and MAE for every (user, item) pair in *eval_df*."""
    predictions = []
    actuals = []

    for row in eval_df.itertuples(index=False):
        pred = model.predict(int(row.userId), int(row.movieId))
        predictions.append(pred)
        actuals.append(float(row.rating))

    predictions_arr = np.array(predictions)
    actuals_arr = np.array(actuals)

    rmse = float(np.sqrt(np.mean((predictions_arr - actuals_arr) ** 2)))
    mae = float(np.mean(np.abs(predictions_arr - actuals_arr)))

    return {
        "rmse": rmse,
        "mae": mae,
        "num_predictions": len(predictions_arr),
    }


def evaluate_recommendations(
    model: Recommender,
    eval_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict:
    """Compute ranking metrics against held-out data."""
    precisions = []
    recalls = []
    ndcgs = []
    hitrates = []

    for uid, group in eval_df.groupby("userId"):
        relevant = set(group.loc[group["rating"] >= relevance_threshold, "movieId"].astype(int))
        if len(relevant) == 0:
            continue

        recs = model.recommend(int(uid), n=k)
        rec_items = [iid for iid, _ in recs]

        hits = [1.0 if iid in relevant else 0.0 for iid in rec_items]
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(relevant))
        hitrates.append(1.0 if sum(hits) > 0 else 0.0)

        dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "hit_rate_at_k": float(np.mean(hitrates)),
        "k": k,
        "num_users_evaluated": len(precisions),
    }


# ------------------------------------------------------------------
# Sampled recommendations
# ------------------------------------------------------------------

def sample_recommendations(
    model: Recommender,
    urm: sp.csr_matrix,
    n_users: int = 5,
    n_recs: int = 10,
    seed: int = 42,
) -> dict[int, list[tuple[int, float]]]:
    """Generate top-n recommendations for a random sample of users."""
    rng = np.random.default_rng(seed)
    n_total_users = urm.shape[0]
    sampled = rng.choice(n_total_users, size=min(n_users, n_total_users), replace=False)

    results: dict[int, list[tuple[int, float]]] = {}
    for uid in sampled:
        results[int(uid)] = model.recommend(int(uid), n=n_recs)
    return results


# ------------------------------------------------------------------
# Full evaluation run
# ------------------------------------------------------------------

def run_evaluation(
    strategies: list[str] | None = None,
    n_sample_users: int = 5,
    n_recs: int = 10,
    checkpoint_dir: str | path.Path | None = None,
    force_refit: bool = False,
) -> dict[str, dict]:
    """Evaluate each strategy and use persisted checkpoints when available."""
    from .fit import default_checkpoint_dir, fit_recommender
    from .load import load_mapping, load_ratings_splits, load_user_item_matrix
    from .recommender_registry import registry
    from .transform import OUT_DIR

    out = path.Path(OUT_DIR)

    urm = load_user_item_matrix(out)
    _train_df, val_df, _test_df = load_ratings_splits(out)

    user_mapping  = load_mapping(out / "user_mapping.csv")
    item_mapping  = load_mapping(out / "item_mapping.csv")
    title_mapping = load_mapping(out / "movie_mapping.csv")

    user_mapping_reverse = {v: k for k, v in user_mapping.items()}
    item_mapping_reverse = {v: k for k, v in item_mapping.items()}

    if strategies is None:
        strategies = registry.names

    checkpoint_root = path.Path(checkpoint_dir) if checkpoint_dir is not None else default_checkpoint_dir()

    all_results: dict[str, dict] = {}

    for name in strategies:
        print(f"\n{'=' * 60}")
        print(f"Evaluating strategy: {name}")   
        print(f"{'=' * 60}")

        spinner = _Spinner(f"{name}")
        t_start = time.perf_counter()
        spinner.start()
        try:
            model = fit_recommender(
                name,
                urm,
                checkpoint_dir=checkpoint_root,
                force_refit=force_refit,
            )
            val_metrics = evaluate_predictions(model, val_df)
            rank_metrics = evaluate_recommendations(model, val_df, k=n_recs)
            recs = sample_recommendations(model, urm, n_users=n_sample_users, n_recs=n_recs)
        finally:
            spinner.stop(time.perf_counter() - t_start)

        all_results[name] = {
            "val": val_metrics,
            "rank": rank_metrics,
            "sample_recs": {str(k): [(iid, sc) for iid, sc in v] for k, v in recs.items()},
        }

    # ---- persist results as CSV for the visualisation endpoint ----

    # 1. Prediction metrics (RMSE / MAE) — one row per strategy
    pred_rows = []
    for name, data in all_results.items():
        pred_rows.append({
            "strategy": name,
            "rmse": data["val"]["rmse"],
            "mae": data["val"]["mae"],
            "num_predictions": data["val"]["num_predictions"],
        })
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out / "eval_prediction_metrics.csv", index=False)

    # 2. Ranking metrics — one row per strategy
    rank_rows = []
    for name, data in all_results.items():
        rank_rows.append({
            "strategy": name,
            "precision_at_k": data["rank"]["precision_at_k"],
            "recall_at_k": data["rank"]["recall_at_k"],
            "ndcg_at_k": data["rank"]["ndcg_at_k"],
            "k": data["rank"]["k"],
            "num_users_evaluated": data["rank"]["num_users_evaluated"],
        })
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(out / "eval_ranking_metrics.csv", index=False)

    # 3. Top-K recommendations with movie titles — one row per (strategy, user, rank)
    rec_rows = []
    for name, data in all_results.items():
        for uid_str, items in data["sample_recs"].items():
            for rank, (iid, score) in enumerate(items, start=1):
                original_iid = item_mapping_reverse.get(iid, iid)
                original_uid = user_mapping_reverse.get(int(uid_str), int(uid_str))
                title = title_mapping.get(original_iid, f"item-{iid}")
                rec_rows.append({
                    "strategy": name,
                    "user_id": original_uid,
                    "rank": rank,
                    "item_id": iid,
                    "title": title,
                    "score": float(score),
                })
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(out / "eval_recommendations.csv", index=False)

    return all_results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _print_results(all_results: dict[str, dict]) -> None:
    """Pretty-print evaluation results to stdout."""
    for name, data in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"Strategy: {name}")
        print(f"{'=' * 60}")

        val = data["val"]
        print(f"\n  RMSE : {val['rmse']:.6f}")
        print(f"  MAE  : {val['mae']:.6f}")

        rank = data["rank"]
        k = rank["k"]
        print(f"\n  Precision@{k} : {rank['precision_at_k']:.6f}")
        print(f"  Recall@{k}    : {rank['recall_at_k']:.6f}")
        print(f"  NDCG@{k}      : {rank['ndcg_at_k']:.6f}")
        print(f"  Users evaluated: {rank['num_users_evaluated']}")


def main():
    results = run_evaluation()
    _print_results(results)


if __name__ == "__main__":
    main()
