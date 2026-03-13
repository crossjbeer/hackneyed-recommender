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
        self.similarity_full: sp.csr_matrix | None = None
        self.similarity_topk: sp.csr_matrix | None = None
        self.abs_similarity_topk: sp.csr_matrix | None = None
        self.urm: sp.csr_matrix | None = None

    def _keep_topk_per_row(self, mat: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """Keep only top-k values per row of a CSR matrix."""
        mat = mat.tocsr()
        data = []
        indices = []
        indptr = [0]

        for i in range(mat.shape[0]):
            start, end = mat.indptr[i], mat.indptr[i + 1]
            row_data = mat.data[start:end]
            row_idx = mat.indices[start:end]

            if len(row_data) > k:
                topk = np.argpartition(row_data, -k)[-k:]
                row_data = row_data[topk]
                row_idx = row_idx[topk]

                # Optional: sort descending for cleaner structure
                order = np.argsort(-row_data)
                row_data = row_data[order]
                row_idx = row_idx[order]

            data.extend(row_data)
            indices.extend(row_idx)
            indptr.append(len(data))

        return sp.csr_matrix(
            (np.array(data), np.array(indices), np.array(indptr)),
            shape=mat.shape,
        )

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm.tocsr()

        norms = np.sqrt(np.asarray(self.urm.power(2).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        urm_normed = self.urm @ sp.diags(1.0 / norms)

        sim = (urm_normed.T @ urm_normed).tocsr()
        sim.setdiag(0.0)
        sim.eliminate_zeros()

        self.similarity_full = sim
        self.similarity_topk = self._keep_topk_per_row(sim, self.k)
        self.abs_similarity_topk = abs(self.similarity_topk)

    def predict(self, user_id: int, item_id: int) -> float:
        user_row = self.urm[user_id]
        rated_items = user_row.indices
        ratings = user_row.data.copy()

        if len(rated_items) == 0:
            return 0.0

        sims = self.similarity_full[item_id, rated_items].toarray().ravel()

        if len(sims) > self.k:
            top_k_idx = np.argpartition(sims, -self.k)[-self.k:]
            sims = sims[top_k_idx]
            ratings = ratings[top_k_idx]

        denom = np.abs(sims).sum()
        if denom == 0:
            return 0.0

        return float(np.dot(sims, ratings) / denom)
    
    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        user_row = self.urm[user_id]

        if user_row.nnz == 0:
            return []

        # Weighted sums for all items at once
        numerators = (user_row @ self.similarity_topk).toarray().ravel()

        # Denominator uses binary "has rated" mask
        user_implicit = sp.csr_matrix(
            (np.ones_like(user_row.data), user_row.indices, [0, len(user_row.indices)]),
            shape=user_row.shape,
        )
        denominators = (user_implicit @ self.abs_similarity_topk).toarray().ravel()

        scores = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators, dtype=float),
            where=denominators != 0,
        )

        # Exclude seen items
        scores[user_row.indices] = -np.inf

        if n >= len(scores):
            top_n_idx = np.argsort(-scores)
        else:
            top_n_idx = np.argpartition(scores, -n)[-n:]
            top_n_idx = top_n_idx[np.argsort(-scores[top_n_idx])]

        return [
            (int(i), float(scores[i]))
            for i in top_n_idx
            if np.isfinite(scores[i])
        ]

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
