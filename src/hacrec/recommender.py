"""Base recommender interface and model-agnostic evaluation utilities."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Recommender(ABC):
    """Base class for recommendation models.

    Subclasses must implement fit() and predict() so that
    evaluate_predictions() can work with any model uniformly.
    """

    @abstractmethod
    def fit(self, urm: sp.csr_matrix) -> None:
        """Train the model on the user-item interaction matrix."""
        ...

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict a rating for a given user-item pair."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the top-n recommended item indices with predicted scores for a user."""
        ...


# ------------------------------------------------------------------
# Model-agnostic evaluation loop (reusable for any Recommender)
# ------------------------------------------------------------------

def evaluate_predictions(model: Recommender, eval_df: pd.DataFrame) -> dict:
    """Compute RMSE and MAE for every (user, item) pair in *eval_df*.

    Works with any Recommender subclass, so the same loop can be
    reused once matrix factorisation is implemented.
    """
    predictions = []
    actuals = []

    for row in eval_df.itertuples(index=False):
        pred = model.predict(int(row.userId), int(row.movieId))
        predictions.append(pred)
        actuals.append(float(row.rating))

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    mae = float(np.mean(np.abs(predictions - actuals)))

    return {
        "rmse": rmse,
        "mae": mae,
        "num_predictions": len(predictions),
    }


def evaluate_recommendations(
    model: Recommender,
    eval_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict:
    """Compute ranking metrics (Precision@K, Recall@K, NDCG@K) against held-out data.

    An item is considered *relevant* for a user if its rating in eval_df
    is >= relevance_threshold.
    """
    precisions = []
    recalls = []
    ndcgs = []

    for uid, group in eval_df.groupby("userId"):
        relevant = set(group.loc[group["rating"] >= relevance_threshold, "movieId"].astype(int))
        if len(relevant) == 0:
            continue

        recs = model.recommend(int(uid), n=k)
        print(recs)
        rec_items = [iid for iid, _ in recs]

        hits = [1.0 if iid in relevant else 0.0 for iid in rec_items]
        print(hits)

        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(relevant))

        # DCG / IDCG
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "k": k,
        "num_users_evaluated": len(precisions),
    }
