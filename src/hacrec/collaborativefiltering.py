"""Item-item Cosine Collaborative Filtering."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib as path

from .transform import OUT_DIR


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


class ItemBasedCF(Recommender):
    """Item-based collaborative filtering using cosine similarity.

    After fit(), the model holds an item-item cosine similarity matrix.
    Predictions are weighted averages of the user's known ratings,
    weighted by similarity to the target item (top-k neighbors).
    """

    def __init__(self, k: int = 50):
        self.k = k
        self.similarity: sp.csr_matrix | None = None
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm

        # Column-normalise URM so that dot products yield cosine similarity.
        norms = np.sqrt(np.array(urm.power(2).sum(axis=0)).flatten())
        norms[norms == 0] = 1.0
        urm_normed = urm @ sp.diags(1.0 / norms)

        self.similarity = (urm_normed.T @ urm_normed).tocsr()
        # print(f"Computed item-item similarity matrix: {self.similarity.shape}")

    def predict(self, user_id: int, item_id: int) -> float:
        user_row = self.urm[user_id]
        rated_items = user_row.indices
        ratings = user_row.data.copy()

        if len(rated_items) == 0:
            return 0.0

        sims = np.array(
            self.similarity[item_id, rated_items].todense()
        ).flatten()

        # Keep only the k most-similar neighbours.
        if len(sims) > self.k:
            top_k_idx = np.argpartition(sims, -self.k)[-self.k:]
            sims = sims[top_k_idx]
            ratings = ratings[top_k_idx]

        denom = np.abs(sims).sum()
        if denom == 0:
            return 0.0

        return float(np.dot(sims, ratings) / denom)

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        num_items = self.urm.shape[1]
        rated_items = set(self.urm[user_id].indices)

        scores = []
        for item_id in range(num_items):
            if item_id in rated_items:
                continue
            score = self.predict(user_id, item_id)
            scores.append((item_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


# ------------------------------------------------------------------
# Model-agnostic evaluation loop (reusable for matrix factorisation)
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
        rec_items = [iid for iid, _ in recs]

        hits = [1.0 if iid in relevant else 0.0 for iid in rec_items]

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


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    out = path.Path(OUT_DIR)

    urm = sp.load_npz(out / "user_item_matrix.npz")
    val_df = pd.read_csv(out / "val_ratings.csv")
    test_df = pd.read_csv(out / "test_ratings.csv")

    print(f"URM shape: {urm.shape}")

    model = ItemBasedCF(k=50)
    model.fit(urm)

    print(f"\nValidation set ({len(val_df)} pairs):")
    val_metrics = evaluate_predictions(model, val_df)
    print(f"  RMSE : {val_metrics['rmse']:.4f}")
    print(f"  MAE  : {val_metrics['mae']:.4f}")

    print(f"\nTest set ({len(test_df)} pairs):")
    test_metrics = evaluate_predictions(model, test_df)
    print(f"  RMSE : {test_metrics['rmse']:.4f}")
    print(f"  MAE  : {test_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
