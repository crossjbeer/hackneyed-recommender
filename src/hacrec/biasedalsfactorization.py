"""Biased Alternating Least Squares (ALS) Matrix Factorization.

Model: r_ui ≈ mu + b_u + b_i + p_u^T q_i

Training:
  1. Estimate global mean mu from all observed ratings.
  2. Alternate bias estimation: b_u and b_i from observed residuals.
  3. Build residual matrix and ALS-factorize it.
  4. Predict = baseline + dot product, clipped to rating range.
"""

import json
import pathlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .recommender import Recommender, evaluate_predictions
from .transform import OUT_DIR


class BiasedALSFactorization(Recommender):
    """ALS matrix factorization with user and item biases.

    Factorises the residual matrix (after removing the baseline
    mu + b_u + b_i) as U @ V^T where U is (n_users, n_factors)
    and V is (n_items, n_factors).

    Weighted per-row regularisation (lambda_ * nnz) scales the
    regularisation term with the number of observed ratings, which
    prevents over-regularising sparse rows.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 20,
        lambda_: float = 0.1,
        min_rating: float = 0.5,
        max_rating: float = 5.0,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.global_mean: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm
        n_users, n_items = urm.shape
        rng = np.random.default_rng(42)

        # --- 1. Global mean -------------------------------------------
        rows, cols = urm.nonzero()
        ratings = np.asarray(urm[rows, cols]).ravel()
        self.global_mean = float(ratings.mean())

        # --- 2. Alternating bias estimation ----------------------------
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        urm_csc = urm.tocsc()

        for _ in range(10):
            for u in range(n_users):
                row = urm.getrow(u)
                idx = row.indices
                if len(idx) == 0:
                    continue
                r = row.data
                self.user_bias[u] = float(
                    np.mean(r - self.global_mean - self.item_bias[idx])
                )

            for i in range(n_items):
                col = urm_csc.getcol(i)
                idx = col.indices
                if len(idx) == 0:
                    continue
                r = col.data
                self.item_bias[i] = float(
                    np.mean(r - self.global_mean - self.user_bias[idx])
                )

        # --- 3. Initialise latent factors ------------------------------
        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors))

        # --- 4. ALS on residuals --------------------------------------
        self.loss_history: list[float] = []

        for _ in range(self.n_iterations):
            self.user_factors = self._solve_users(urm)
            self.item_factors = self._solve_items(urm_csc)

            loss = self._compute_loss(urm)
            self.loss_history.append(loss)

    # ------------------------------------------------------------------
    # ALS solvers
    # ------------------------------------------------------------------

    def _solve_users(self, urm: sp.csr_matrix) -> np.ndarray:
        """Fix item factors; solve for each user vector."""
        n_users = urm.shape[0]
        result = np.zeros((n_users, self.n_factors))

        for u in range(n_users):
            row = urm.getrow(u)
            idx = row.indices
            if len(idx) == 0:
                continue
            F = self.item_factors[idx]                          # (nnz, k)
            residuals = (
                row.data
                - self.global_mean
                - self.user_bias[u]
                - self.item_bias[idx]
            )
            reg = self.lambda_ * len(idx) * np.eye(self.n_factors)
            A = F.T @ F + reg                                   # (k, k)
            b = F.T @ residuals                                 # (k,)
            result[u] = np.linalg.solve(A, b)

        return result

    def _solve_items(self, urm_csc: sp.csc_matrix) -> np.ndarray:
        """Fix user factors; solve for each item vector."""
        n_items = urm_csc.shape[1]
        result = np.zeros((n_items, self.n_factors))

        for i in range(n_items):
            col = urm_csc.getcol(i)
            idx = col.indices                                   # user indices
            if len(idx) == 0:
                continue
            F = self.user_factors[idx]                         # (nnz, k)
            residuals = (
                col.data
                - self.global_mean
                - self.user_bias[idx]
                - self.item_bias[i]
            )
            reg = self.lambda_ * len(idx) * np.eye(self.n_factors)
            A = F.T @ F + reg                                  # (k, k)
            b = F.T @ residuals                                # (k,)
            result[i] = np.linalg.solve(A, b)

        return result

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, urm: sp.csr_matrix) -> float:
        """MSE on residuals (observed entries) + L2 regularisation."""
        rows, cols = urm.nonzero()
        baseline = self.global_mean + self.user_bias[rows] + self.item_bias[cols]
        latent = np.sum(self.user_factors[rows] * self.item_factors[cols], axis=1)
        errors = np.asarray(urm[rows, cols]).ravel() - baseline - latent
        mse = float(np.mean(errors ** 2))
        reg = self.lambda_ * (
            np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
        )
        return mse + reg

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, user_id: int, item_id: int) -> float:
        pred = self.global_mean
        if self.user_bias is not None and user_id < len(self.user_bias):
            pred += self.user_bias[user_id]
        if self.item_bias is not None and item_id < len(self.item_bias):
            pred += self.item_bias[item_id]
        if (
            self.user_factors is not None
            and self.item_factors is not None
            and user_id < len(self.user_factors)
            and item_id < len(self.item_factors)
        ):
            pred += float(self.user_factors[user_id] @ self.item_factors[item_id])
        return float(np.clip(pred, self.min_rating, self.max_rating))

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        scores = (
            self.global_mean
            + self.user_bias[user_id]
            + self.item_bias
            + self.user_factors[user_id] @ self.item_factors.T
        )
        scores = np.clip(scores, self.min_rating, self.max_rating)
        rated_items = set(self.urm[user_id].indices) if self.urm is not None else set()
        candidate_idx = [i for i in range(len(scores)) if i not in rated_items]
        top_idx = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:n]
        return [(i, float(scores[i])) for i in top_idx]

    def __str__(self) -> str:
        return "Biased ALS Matrix Factorization"

    def __repr__(self) -> str:
        return (
            "BiasedALSFactorization("
            f"n_factors={self.n_factors}, "
            f"n_iterations={self.n_iterations}, "
            f"lambda_={self.lambda_}, "
            f"min_rating={self.min_rating}, "
            f"max_rating={self.max_rating}"
            ")"
        )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    out = pathlib.Path(OUT_DIR)

    urm = sp.load_npz(out / "user_item_matrix.npz")
    val_df = pd.read_csv(out / "val_ratings.csv")

    print(f"URM shape: {urm.shape}")

    model = BiasedALSFactorization(n_factors=20, n_iterations=40, lambda_=0.1)
    model.fit(urm)

    loss_path = out / "biased_als_loss_history.json"
    with open(loss_path, "w") as f:
        json.dump(model.loss_history, f)
    print(f"Saved loss history to {loss_path}")

    print(f"\nValidation set ({len(val_df)} pairs):")
    val_metrics = evaluate_predictions(model, val_df)
    print(f"  RMSE : {val_metrics['rmse']:.4f}")
    print(f"  MAE  : {val_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
