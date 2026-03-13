"""Biased Item-item Collaborative Filtering with baseline-corrected residuals.

Model: pred_ui = (mu + b_u + b_i) + CF_residual(u, i)

Training:
  1. Estimate global mean mu plus regularised user/item biases b_u, b_i.
  2. Build a residual URM:  r~_ui = r_ui - (mu + b_u + b_i)  for every
     observed rating.
  3. Compute item-item cosine similarity on the residual URM (top-k kept
     per item for fast recommendation).
  4. Predict: baseline + weighted average of the user's residuals for
     similar items.  Clip the result to [min_rating, max_rating].
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib as path

from .recommender import Recommender, evaluate_predictions, evaluate_recommendations
from .transform import OUT_DIR


class BiasedCollaborativeCF(Recommender):
    """Item-based CF on baseline-corrected residuals.

    Biases are estimated with regularised alternating least squares so that
    sparse users/items are shrunk toward the global mean.  The item-item
    cosine similarity is then computed on the residual interaction matrix,
    making neighbourhood weights independent of rating scale offsets.
    """

    def __init__(
        self,
        k: int = 50,
        reg: float = 10.0,
        n_iterations: int = 10,
        min_rating: float = 0.5,
        max_rating: float = 5.0,
    ):
        self.k = k
        self.reg = reg
        self.n_iterations = n_iterations
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.global_mean: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.similarity_full: sp.csr_matrix | None = None
        self.similarity_topk: sp.csr_matrix | None = None
        self.abs_similarity_topk: sp.csr_matrix | None = None
        self.residual_urm: sp.csr_matrix | None = None
        self.urm: sp.csr_matrix | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_biases(self, urm: sp.csr_matrix) -> None:
        """Regularised ALS bias estimation: mu, b_u, b_i."""
        n_users, n_items = urm.shape
        self.global_mean = float(urm.data.mean()) if urm.nnz > 0 else 0.0
        self.user_bias = np.zeros(n_users, dtype=float)
        self.item_bias = np.zeros(n_items, dtype=float)

        urm_csc = urm.tocsc()

        for _ in range(self.n_iterations):
            for u in range(n_users):
                row = urm.getrow(u)
                if row.nnz == 0:
                    continue
                residual = row.data - self.global_mean - self.item_bias[row.indices]
                self.user_bias[u] = float(residual.sum() / (self.reg + row.nnz))

            for i in range(n_items):
                col = urm_csc.getcol(i)
                if col.nnz == 0:
                    continue
                residual = col.data - self.global_mean - self.user_bias[col.indices]
                self.item_bias[i] = float(residual.sum() / (self.reg + col.nnz))

    def _build_residual_urm(self, urm: sp.csr_matrix) -> sp.csr_matrix:
        """Return a sparse matrix of (r_ui - baseline_ui) matching urm's sparsity."""
        rows, cols = urm.nonzero()
        baselines = self.global_mean + self.user_bias[rows] + self.item_bias[cols]
        residuals = np.asarray(urm[rows, cols]).ravel().astype(float) - baselines
        return sp.csr_matrix((residuals, (rows, cols)), shape=urm.shape)

    def _keep_topk_per_row(self, mat: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """Keep only the top-k values per row of a CSR matrix."""
        mat = mat.tocsr()
        data: list = []
        indices: list = []
        indptr = [0]

        for i in range(mat.shape[0]):
            start, end = mat.indptr[i], mat.indptr[i + 1]
            row_data = mat.data[start:end]
            row_idx = mat.indices[start:end]

            if len(row_data) > k:
                topk = np.argpartition(row_data, -k)[-k:]
                row_data = row_data[topk]
                row_idx = row_idx[topk]
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

    # ------------------------------------------------------------------
    # Recommender interface
    # ------------------------------------------------------------------

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm.tocsr()

        # 1. Estimate global mean, user biases, item biases
        self._estimate_biases(self.urm)

        # 2. Build residual URM
        self.residual_urm = self._build_residual_urm(self.urm)

        # 3. Item-item cosine similarity on residuals
        norms = np.sqrt(np.asarray(self.residual_urm.power(2).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        res_normed = self.residual_urm @ sp.diags(1.0 / norms)

        sim = (res_normed.T @ res_normed).tocsr()
        sim.setdiag(0.0)
        sim.eliminate_zeros()

        self.similarity_full = sim
        self.similarity_topk = self._keep_topk_per_row(sim, self.k)
        self.abs_similarity_topk = abs(self.similarity_topk)

    def predict(self, user_id: int, item_id: int) -> float:
        bu = self.user_bias[user_id] if user_id < len(self.user_bias) else 0.0
        bi = self.item_bias[item_id] if item_id < len(self.item_bias) else 0.0
        baseline = self.global_mean + bu + bi

        user_res_row = self.residual_urm[user_id]
        rated_items = user_res_row.indices
        residuals = user_res_row.data

        if len(rated_items) == 0:
            return float(np.clip(baseline, self.min_rating, self.max_rating))

        sims = self.similarity_full[item_id, rated_items].toarray().ravel()

        if len(sims) > self.k:
            top_k_idx = np.argpartition(sims, -self.k)[-self.k:]
            sims = sims[top_k_idx]
            residuals = residuals[top_k_idx]

        denom = np.abs(sims).sum()
        if denom == 0:
            return float(np.clip(baseline, self.min_rating, self.max_rating))

        residual_pred = float(np.dot(sims, residuals) / denom)
        return float(np.clip(baseline + residual_pred, self.min_rating, self.max_rating))

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        user_res_row = self.residual_urm[user_id]

        # Baseline scores for every item
        bu = self.user_bias[user_id] if user_id < len(self.user_bias) else 0.0
        baseline_scores = self.global_mean + bu + self.item_bias  # (n_items,)

        if user_res_row.nnz == 0:
            scores = baseline_scores.copy()
        else:
            # Vectorised residual CF component
            numerators = (user_res_row @ self.similarity_topk).toarray().ravel()

            user_implicit = sp.csr_matrix(
                (np.ones_like(user_res_row.data), user_res_row.indices, [0, len(user_res_row.indices)]),
                shape=user_res_row.shape,
            )
            denominators = (user_implicit @ self.abs_similarity_topk).toarray().ravel()

            cf_residuals = np.divide(
                numerators,
                denominators,
                out=np.zeros_like(numerators, dtype=float),
                where=denominators != 0,
            )
            scores = baseline_scores + cf_residuals

        # Exclude already-rated items
        scores[self.urm[user_id].indices] = -np.inf

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

    model = BiasedCollaborativeCF(k=50, reg=10.0)
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
