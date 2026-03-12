"""Unified evaluation endpoint for all recommender strategies."""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib as path

from .collaborativefiltering import ItemBasedCF, Recommender, evaluate_predictions, evaluate_recommendations
from .baselines import (
    GlobalMeanBaseline,
    UserMeanBaseline,
    ItemMeanBaseline,
    UserItemBiasBaseline,
    MostPopularBaseline,
)
from .factorization import ALSFactorization
from .transform import DATA_DIR, MOVIELENS_DIR, OUT_DIR, load_mapping 


# ------------------------------------------------------------------
# Strategy registry
# ------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[Recommender]] = {
    "item-cf": ItemBasedCF,
    "als": ALSFactorization,
    "global-mean": GlobalMeanBaseline,
    "user-mean": UserMeanBaseline,
    "item-mean": ItemMeanBaseline,
    "user-item-bias": UserItemBiasBaseline,
    "most-popular": MostPopularBaseline,
}

DEFAULT_PARAMS: dict[str, dict] = {
    "item-cf": {"k": 50},
    "als": {"n_factors": 65, "n_iterations": 40, "lambda_": 0.1},
    "global-mean": {},
    "user-mean": {},
    "item-mean": {},
    "user-item-bias": {"reg": 10.0, "n_iterations": 10},
    "most-popular": {},
}


def build_model(strategy: str, **overrides) -> Recommender:
    """Instantiate a registered recommender strategy.

    Any keyword arguments override the defaults for that strategy.
    """
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    params = {**DEFAULT_PARAMS.get(strategy, {}), **overrides}
    return STRATEGY_REGISTRY[strategy](**params)


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
    n_recs: int = 5,
) -> dict[str, dict]:
    """Train each strategy, evaluate on val/test sets, and sample recommendations.

    Returns a dict keyed by strategy name with metrics and sample recs.
    """
    out = path.Path(OUT_DIR)

    urm = sp.load_npz(out / "user_item_matrix.npz")
    val_df = pd.read_csv(out / "val_ratings.csv")
    test_df = pd.read_csv(out / "test_ratings.csv")

    user_mapping = load_mapping(out / "user_mapping.csv")
    item_mapping = load_mapping(out / "item_mapping.csv")
    title_mapping = load_mapping(out / "movie_mapping.csv")

    user_mapping_reverse = {v: k for k, v in user_mapping.items()}
    item_mapping_reverse = {v: k for k, v in item_mapping.items()}

    if strategies is None:
        strategies = list(STRATEGY_REGISTRY.keys())

    all_results: dict[str, dict] = {}

    for name in strategies:
        print(f"\n{'=' * 60}")
        print(f"Evaluating strategy: {name}")   
        print(f"{'=' * 60}")
        
        model = build_model(name)
        model.fit(urm)

        val_metrics = evaluate_predictions(model, val_df)
        rank_metrics = evaluate_recommendations(model, val_df, k=n_recs)
        recs = sample_recommendations(model, urm, n_users=n_sample_users, n_recs=n_recs)

        all_results[name] = {
            "val": val_metrics,
            "rank": rank_metrics,
            "sample_recs": {str(k): [(iid, sc) for iid, sc in v] for k, v in recs.items()},
        }

        # Persist ALS loss history when applicable
        if isinstance(model, ALSFactorization) and hasattr(model, "loss_history"):
            loss_path = out / "als_loss_history.json"
            with open(loss_path, "w") as f:
                json.dump(model.loss_history, f)

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
                title = title_mapping.get(original_iid, f"item-{iid}")
                rec_rows.append({
                    "strategy": name,
                    "user_id": uid_str,
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
        print(f"\n  RMSE : {val['rmse']:.4f}")
        print(f"  MAE  : {val['mae']:.4f}")

        rank = data["rank"]
        k = rank["k"]
        print(f"\n  Precision@{k} : {rank['precision_at_k']:.4f}")
        print(f"  Recall@{k}    : {rank['recall_at_k']:.4f}")
        print(f"  NDCG@{k}      : {rank['ndcg_at_k']:.4f}")
        print(f"  Users evaluated: {rank['num_users_evaluated']}")


def main():
    results = run_evaluation()
    _print_results(results)


if __name__ == "__main__":
    main()
