"""Bayesian Personalized Ranking (BPR) Matrix Factorization.

BPR optimises a pairwise ranking objective (Rendle et al., 2009) rather than
predicting absolute ratings.  For each observed (user, positive-item) pair it
samples a random unobserved (negative) item and pushes the model to rank the
positive item above the negative one via SGD on the BPR-OPT criterion:

    BPR-OPT = sum_( u,i,j ) ln σ( x_uij ) - λ ||Θ||²

where x_uij = U[u] · V[i] - U[u] · V[j] is the preference score difference
and Θ = {U, V} are the latent factor matrices.
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib as path

from .recommender import Recommender, evaluate_recommendations
from .transform import OUT_DIR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class BPRFactorization(Recommender):
    """Bayesian Personalized Ranking matrix factorization.

    Factorises the user-item matrix R ≈ U @ V^T where U is (n_users, n_factors)
    and V is (n_items, n_factors).  Training uses pairwise SGD: for every
    observed interaction (u, i) a random unobserved item j is drawn and the
    parameters are updated to increase the predicted score difference x_uij.

    Parameters
    ----------
    n_factors:    Dimensionality of the latent factor space.
    n_epochs:     Number of full passes over all observed interactions.
    lr:           SGD learning rate.
    lambda_:      L2 regularisation strength applied to both U and V.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr: float = 0.05,
        lambda_: float = 0.01,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.lambda_ = lambda_
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        """Train on the user-item matrix using BPR-SGD."""
        self.urm = urm
        n_users, n_items = urm.shape
        rng = np.random.default_rng(42)

        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors))

        # Precompute per-user positive item lists for fast sampling
        pos_items_per_user = [
            urm.getrow(u).indices.tolist() for u in range(n_users)
        ]
        all_items = np.arange(n_items)

        self.loss_history: list[float] = []

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_samples = 0

            for u in rng.permutation(n_users):
                pos_items = pos_items_per_user[u]
                if len(pos_items) == 0:
                    continue

                # One SGD step per positive item the user has rated
                for i in pos_items:
                    # Sample a negative item not observed by u
                    neg_mask = np.ones(n_items, dtype=bool)
                    neg_mask[pos_items] = False
                    neg_candidates = all_items[neg_mask]
                    j = int(rng.choice(neg_candidates))

                    # Preference score difference x_uij = U[u]·(V[i] - V[j])
                    diff = self.item_factors[i] - self.item_factors[j]
                    x_uij = float(self.user_factors[u] @ diff)
                    sigma = float(_sigmoid(-x_uij))   # gradient coefficient

                    # Gradient ascent on BPR-OPT
                    grad_u = sigma * diff - self.lambda_ * self.user_factors[u]
                    grad_vi = sigma * self.user_factors[u] - self.lambda_ * self.item_factors[i]
                    grad_vj = -sigma * self.user_factors[u] - self.lambda_ * self.item_factors[j]

                    self.user_factors[u] += self.lr * grad_u
                    self.item_factors[i] += self.lr * grad_vi
                    self.item_factors[j] += self.lr * grad_vj

                    epoch_loss += -np.log(_sigmoid(x_uij) + 1e-10)
                    n_samples += 1

            avg_loss = float(epoch_loss / max(n_samples, 1))
            self.loss_history.append(avg_loss)
            # print(f"  Epoch {epoch + 1}/{self.n_epochs}  loss={avg_loss:.4f}")

    def predict(self, user_id: int, item_id: int) -> float:
        """Return the latent-factor dot-product score for a user-item pair."""
        return float(self.user_factors[user_id] @ self.item_factors[item_id])

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the top-n unrated items ranked by predicted score."""
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

    model = BPRFactorization(n_factors=50, n_epochs=20, lr=0.05, lambda_=0.01)
    model.fit(urm)

    # Persist loss history
    loss_path = out / "bpr_loss_history.json"
    with open(loss_path, "w") as f:
        json.dump(model.loss_history, f)
    print(f"Saved BPR loss history to {loss_path}")

    print(f"\nValidation set — ranking metrics (k=10):")
    val_metrics = evaluate_recommendations(model, val_df, k=10)
    print(f"  Precision@10 : {val_metrics['precision_at_k']:.4f}")
    print(f"  Recall@10    : {val_metrics['recall_at_k']:.4f}")
    print(f"  NDCG@10      : {val_metrics['ndcg_at_k']:.4f}")
    print(f"  HitRate@10   : {val_metrics['hit_rate_at_k']:.4f}")


if __name__ == "__main__":
    main()
