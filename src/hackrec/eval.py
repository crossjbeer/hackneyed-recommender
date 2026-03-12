"""Unified evaluation endpoint for all recommender strategies."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib as path

from .collaborativefiltering import ItemBasedCF, Recommender, evaluate_predictions, evaluate_recommendations
from .factorization import ALSFactorization
from .transform import DATA_DIR, MOVIELENS_DIR, OUT_DIR, load_mapping 


# ------------------------------------------------------------------
# Strategy registry
# ------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[Recommender]] = {
    "item-cf": ItemBasedCF,
    "als": ALSFactorization,
}

DEFAULT_PARAMS: dict[str, dict] = {
    "item-cf": {"k": 50},
    "als": {"n_factors": 65, "n_iterations": 40, "lambda_": 0.1},
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

    print(f"URM shape: {urm.shape}")

    if strategies is None:
        strategies = list(STRATEGY_REGISTRY.keys())

    all_results: dict[str, dict] = {}

    for name in strategies:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {name}")
        print(f"{'=' * 60}")

        model = build_model(name)
        model.fit(urm)

        # --- validation metrics ---
        print(f"\nValidation set ({len(val_df)} pairs):")
        val_metrics = evaluate_predictions(model, val_df)
        print(f"  RMSE : {val_metrics['rmse']:.4f}")
        print(f"  MAE  : {val_metrics['mae']:.4f}")

        # --- test metrics ---
        # print(f"\nTest set ({len(test_df)} pairs):")
        # test_metrics = evaluate_predictions(model, test_df)
        # print(f"  RMSE : {test_metrics['rmse']:.4f}")
        # print(f"  MAE  : {test_metrics['mae']:.4f}")

        # --- ranking metrics ---
        print(f"\nRanking metrics (top-{n_recs}):")
        rank_metrics = evaluate_recommendations(model, val_df, k=n_recs)
        print(f"  Precision@{n_recs} : {rank_metrics['precision_at_k']:.4f}")
        print(f"  Recall@{n_recs}    : {rank_metrics['recall_at_k']:.4f}")
        print(f"  NDCG@{n_recs}      : {rank_metrics['ndcg_at_k']:.4f}")
        print(f"  Users evaluated: {rank_metrics['num_users_evaluated']}")

        # --- sampled recommendations ---
        print(f"\nSample recommendations ({n_sample_users} users, top-{n_recs}):")
        recs = sample_recommendations(model, urm, n_users=n_sample_users, n_recs=n_recs)
        for uid, items in recs.items():
            top_items = ", ".join(f"{title_mapping[item_mapping_reverse[iid]]} ({score:.3f})" for iid, score in items)
            print(f"  User {user_mapping_reverse[uid]}: {top_items}")

        all_results[name] = {
            "val": val_metrics,
            # "test": test_metrics,
            "rank": rank_metrics,
            "sample_recs": recs,
        }

    return all_results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    run_evaluation()


if __name__ == "__main__":
    main()
