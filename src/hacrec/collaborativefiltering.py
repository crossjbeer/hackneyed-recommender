"""Item-item Cosine Collaborative Filtering."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib as path

from .recommender import Recommender, evaluate_predictions, evaluate_recommendations
from .transform import OUT_DIR


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
    val_rank_metrics = evaluate_recommendations(model, val_df, k=10, relevance_threshold=4.0)
    print(f"  RMSE : {val_metrics['rmse']:.4f}")
    print(f"  MAE  : {val_metrics['mae']:.4f}")
    print(f"  Precision@10 : {val_rank_metrics['precision_at_k']:.4f}")
    print(f"  Recall@10    : {val_rank_metrics['recall_at_k']:.4f}")
    print(f"  NDCG@10      : {val_rank_metrics['ndcg_at_k']:.4f}")

    # print(f"\nTest set ({len(test_df)} pairs):")
    # test_metrics = evaluate_predictions(model, test_df)
    # print(f"  RMSE : {test_metrics['rmse']:.4f}")
    # print(f"  MAE  : {test_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
