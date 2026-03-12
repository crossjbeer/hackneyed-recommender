"""Matrix factorization (ALS) for rating prediction."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib as path

from .collaborativefiltering import Recommender, evaluate_predictions
from .transform import OUT_DIR


class ALSFactorization(Recommender):
    """Alternating Least Squares matrix factorization.

    Factorises the user-item matrix R ≈ U @ V^T where U is (n_users, n_factors)
    and V is (n_items, n_factors).  Regularisation (lambda_) prevents overfitting.
    """

    def __init__(self, n_factors: int = 50, n_iterations: int = 20, lambda_: float = 0.1):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm
        n_users, n_items = urm.shape
        rng = np.random.default_rng(42)

        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors))

        urm_csc = urm.tocsc()

        for iteration in range(self.n_iterations):
            # Fix items, solve for users
            self.user_factors = self._solve(urm, self.item_factors)
            # Fix users, solve for items
            self.item_factors = self._solve(urm_csc.T.tocsr(), self.user_factors)

            # Training loss (evaluated on nonzero entries only)
            loss = self._compute_loss(urm)
            # print(f"  Iteration {iteration + 1}/{self.n_iterations}  loss={loss:.4f}")

        # print(f"ALS fit complete — factors={self.n_factors}, shape U={self.user_factors.shape}, V={self.item_factors.shape}")

    def _solve(self, ratings: sp.csr_matrix, fixed_factors: np.ndarray) -> np.ndarray:
        """Solve for one factor matrix while the other is held fixed.

        For each row i of *ratings* we solve the ridge-regression problem:
            argmin_x  ||R_i - x @ F^T||^2  +  λ ||x||^2
        where F = fixed_factors and R_i are the nonzero entries in row i.
        """
        n_rows = ratings.shape[0]
        n_factors = fixed_factors.shape[1]
        result = np.zeros((n_rows, n_factors))
        reg = self.lambda_ * np.eye(n_factors)

        for i in range(n_rows):
            row = ratings.getrow(i)
            idx = row.indices
            if len(idx) == 0:
                continue
            F_i = fixed_factors[idx]                 # (nnz, k)
            R_i = row.data                           # (nnz,)
            A = F_i.T @ F_i + reg                    # (k, k)
            b = F_i.T @ R_i                          # (k,)
            result[i] = np.linalg.solve(A, b)

        return result

    def _compute_loss(self, urm: sp.csr_matrix) -> float:
        """MSE on observed entries + L2 regularisation."""
        rows, cols = urm.nonzero()
        preds = np.sum(self.user_factors[rows] * self.item_factors[cols], axis=1)
        errors = np.array(urm[rows, cols]).flatten() - preds
        mse = np.mean(errors ** 2)
        reg = self.lambda_ * (
            np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
        )
        return mse + reg

    def predict(self, user_id: int, item_id: int) -> float:
        return float(self.user_factors[user_id] @ self.item_factors[item_id])

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        scores = self.user_factors[user_id] @ self.item_factors.T
        rated_items = set(self.urm[user_id].indices) if self.urm is not None else set()
        candidate_idx = [i for i in range(len(scores)) if i not in rated_items]
        top_idx = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:n]
        return [(i, float(scores[i])) for i in top_idx]


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    out = path.Path(OUT_DIR)

    urm = sp.load_npz(out / "user_item_matrix.npz")
    val_df = pd.read_csv(out / "val_ratings.csv")
    test_df = pd.read_csv(out / "test_ratings.csv")

    print(f"URM shape: {urm.shape}")

    model = ALSFactorization(n_factors=65, n_iterations=40, lambda_=0.1)
    model.fit(urm)

    print(f"\nValidation set ({len(val_df)} pairs):")
    val_metrics = evaluate_predictions(model, val_df)
    print(f"  RMSE : {val_metrics['rmse']:.4f}")
    print(f"  MAE  : {val_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()

