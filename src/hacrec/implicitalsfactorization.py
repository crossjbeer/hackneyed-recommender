"""Implicit Alternating Least Squares (iALS) factorization.

Based on Hu, Koren & Volinsky (2008) "Collaborative Filtering for Implicit
Feedback Datasets".

Rather than predicting explicit star ratings, iALS models a binary *preference*
signal and a graded *confidence* that the preference holds:

    p_ui = 1  if the user has rated item i  (any observed interaction)
    c_ui = 1 + α · r_ui                      (confidence; α scales rating magnitude)

The model minimises the weighted reconstruction error over ALL (u, i) pairs:

    min_{U,V}  Σ_{u,i} c_ui (p_ui − u_u · v_i)² + λ (‖U‖² + ‖V‖²)

Because p_ui = 0 for unobserved pairs, the unobserved term acts as a
uniform soft push towards zero — preventing the model from trivially
recommending everything.

Closed-form ALS solution
------------------------
For user u, fix V and differentiate w.r.t. u_u:

    u_u = (V^T C^u V + λI)^{-1} V^T C^u p_u

Exploiting sparsity via the identity:

    V^T C^u V = V^T V  +  V^T (C^u − I) V
                └─────┘   └──────────────────┘
             precomputed    sparse: only nnz(u) terms

    V^T C^u p_u = Σ_{i: r_ui>0} c_ui · v_i        (p_ui = 1 for observed)

The same update applies symmetrically to item factors (transpose the URM).
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib as path

from .recommender import Recommender, evaluate_recommendations
from .transform import OUT_DIR


class ImplicitALSFactorizer(Recommender):
    """iALS: Implicit ALS from Hu, Koren & Volinsky (2008).

    Parameters
    ----------
    n_factors : int
        Dimensionality of the latent factor space.
    n_iterations : int
        Number of ALS sweeps (one sweep = update all users then all items).
    lambda_ : float
        L2 regularisation strength.
    alpha : float
        Confidence scaling factor.  c_ui = 1 + alpha * r_ui.  The original
        paper uses ~40 for play-count data; moderate values (10–40) work well
        for graded explicit ratings repurposed as implicit signals.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 15,
        lambda_: float = 0.1,
        alpha: float = 40.0,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.alpha = alpha
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        """Train iALS on the user-item matrix.

        ``urm`` values are treated as confidence signals, not absolute ratings.
        Any nonzero entry is considered a positive preference (p_ui = 1); the
        rating magnitude only affects the confidence weight c_ui = 1 + α·r_ui.
        """
        self.urm = urm
        n_users, n_items = urm.shape
        rng = np.random.default_rng(42)

        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors))

        # Transpose once; used for the item-factor update (items become rows)
        urm_csc = urm.tocsc()
        urm_items_as_rows = urm_csc.T.tocsr()

        self.loss_history: list[float] = []

        for _iteration in range(self.n_iterations):
            # ---- update user factors (items fixed) -------------------------
            VtV = self.item_factors.T @ self.item_factors       # (k, k)
            self._update_factors(urm, self.item_factors, self.user_factors, VtV)

            # ---- update item factors (users fixed) -------------------------
            UtU = self.user_factors.T @ self.user_factors       # (k, k)
            self._update_factors(urm_items_as_rows, self.user_factors, self.item_factors, UtU)

            self.loss_history.append(self._compute_loss(urm))

    def _update_factors(
        self,
        ratings: sp.csr_matrix,
        fixed: np.ndarray,
        target: np.ndarray,
        FtF: np.ndarray,
    ) -> None:
        """In-place ALS update: solve for each row of *target* while *fixed* is held constant.

        For row i (a user or item), the normal equation is:

            A · x  =  b
            A = F^T F  +  F_i^T diag(α · r_i) F_i  +  λI
            b = F_i^T · c_i                              (c_i = 1 + α r_i)

        where F_i and r_i are the factor rows and ratings at *observed* positions
        for entity i.  F^T F is passed in as *FtF* (precomputed outside the loop).
        """
        k = fixed.shape[1]
        reg = self.lambda_ * np.eye(k)

        for i in range(ratings.shape[0]):
            row = ratings.getrow(i)
            idx = row.indices          # indices of observed interactions
            if len(idx) == 0:
                continue

            r_i = row.data                              # raw ratings at observed positions
            c_i = 1.0 + self.alpha * r_i               # confidence weights  (c_ui = 1 + α r_ui)
            F_i = fixed[idx]                            # (nnz, k)

            # Sparse correction term: F_i^T diag(α r_i) F_i   [= F_i^T diag(c_i - 1) F_i]
            # Added to FtF (the implicit c=1 background already captured there)
            A = FtF + F_i.T @ ((self.alpha * r_i)[:, None] * F_i) + reg

            # RHS: Σ_{observed i} c_ui · v_i   (p_ui = 1 for all observed entries)
            b = F_i.T @ c_i

            target[i] = np.linalg.solve(A, b)

    def _compute_loss(self, urm: sp.csr_matrix) -> float:
        """Observed-entry partial loss (for convergence monitoring).

        Evaluates:  Σ_{observed} c_ui (1 − u_u · v_i)² + λ (‖U‖² + ‖V‖²)

        The unobserved-pair term (p=0, c=1) is omitted for speed; loss
        values are useful for tracking convergence, not for absolute comparison
        with explicit-rating RMSE.
        """
        rows, cols = urm.nonzero()
        r = np.array(urm[rows, cols]).flatten()
        c = 1.0 + self.alpha * r
        preds = np.sum(self.user_factors[rows] * self.item_factors[cols], axis=1)
        weighted_sq_err = np.sum(c * (1.0 - preds) ** 2)   # p_ui = 1 for all observed
        reg = self.lambda_ * (
            np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
        )
        return float(weighted_sq_err / len(rows) + reg)

    def predict(self, user_id: int, item_id: int) -> float:
        """Return the latent-factor preference score u_u · v_i.

        This estimates the binary preference p_ui ∈ {0, 1}, not a star rating.
        RMSE/MAE against explicit ratings carry no meaningful interpretation for
        an implicit model; use ranking metrics (NDCG@K, Precision@K) instead.
        """
        return float(self.user_factors[user_id] @ self.item_factors[item_id])

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the top-n unrated items ranked by preference score."""
        scores = self.user_factors[user_id] @ self.item_factors.T
        rated_items = set(self.urm[user_id].indices) if self.urm is not None else set()
        candidate_idx = [i for i in range(len(scores)) if i not in rated_items]
        top_idx = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:n]
        return [(i, float(scores[i])) for i in top_idx]

    def __str__(self) -> str:
        return "Implicit ALS (iALS)"

    def __repr__(self) -> str:
        return (
            "ImplicitALSFactorizer("
            f"n_factors={self.n_factors}, "
            f"n_iterations={self.n_iterations}, "
            f"lambda_={self.lambda_}, "
            f"alpha={self.alpha}"
            ")"
        )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    out = path.Path(OUT_DIR)

    urm = sp.load_npz(out / "user_item_matrix.npz")
    val_df = pd.read_csv(out / "val_ratings.csv")

    print(f"URM shape: {urm.shape}")

    model = ImplicitALSFactorizer(n_factors=50, n_iterations=15, lambda_=0.1, alpha=40.0)
    model.fit(urm)

    loss_path = out / "ials_loss_history.json"
    with open(loss_path, "w") as f:
        json.dump(model.loss_history, f)
    print(f"Saved iALS loss history to {loss_path}")

    print(f"\nValidation set — ranking metrics (k=10):")
    rank_metrics = evaluate_recommendations(model, val_df, k=10)
    print(f"  Precision@10 : {rank_metrics['precision_at_k']:.4f}")
    print(f"  Recall@10    : {rank_metrics['recall_at_k']:.4f}")
    print(f"  NDCG@10      : {rank_metrics['ndcg_at_k']:.4f}")
    print(f"  HitRate@10   : {rank_metrics['hit_rate_at_k']:.4f}")
    print(f"  Users eval'd : {rank_metrics['num_users_evaluated']}")


if __name__ == "__main__":
    main()
