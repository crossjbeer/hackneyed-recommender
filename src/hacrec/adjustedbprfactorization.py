"""Adjusted Bayesian Personalized Ranking (BPR) Matrix Factorization.

Extends BPRFactorization with two complementary mechanisms to reduce
popularity bias and promote diversity in rankings:

1. **Popularity-biased negative sampling** (training time):
   Negatives are sampled with probability proportional to item popularity
   (interaction count) rather than uniformly.  Popular items appear as
   negatives more often, forcing the model to learn that popularity alone
   does not explain preference.

2. **Popularity penalty** (inference time):
   Recommendation scores are adjusted by subtracting a term proportional
   to the log-popularity of each item:

       adjusted_score(u, i) = U[u] · V[i] - alpha * log(1 + pop(i))

   The ``alpha`` hyper-parameter controls the strength of de-popularisation.
   alpha=0 recovers the vanilla BPR ranking.
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


class AdjustedBPRFactorization(Recommender):
    """Popularity-adjusted Bayesian Personalized Ranking matrix factorization.

    Adds two bias-reduction mechanisms on top of vanilla BPR:

    * **Popularity-biased negative sampling**: during training, unobserved
      items are drawn with probability proportional to their interaction
      count.  This exposes popular items as negatives more frequently,
      discouraging the model from ranking them highly purely because of
      frequency.

    * **Popularity penalty at inference**: final recommendation scores are
      down-weighted by ``alpha * log(1 + popularity(i))``, nudging
      long-tail items up the ranking.

    Parameters
    ----------
    n_factors:
        Dimensionality of the latent factor space.
    n_epochs:
        Number of full passes over all observed interactions.
    lr:
        SGD learning rate.
    lambda_:
        L2 regularisation strength applied to both U and V.
    alpha:
        Popularity-penalty weight applied at recommendation time.
        Larger values favour long-tail items more aggressively.
        Set to 0.0 to disable the penalty (vanilla BPR ranking).
    pop_neg_sampling:
        If True (default), sample negatives proportional to item
        popularity during training.  If False, use uniform sampling
        (identical training behaviour to BPRFactorization).
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr: float = 0.05,
        lambda_: float = 0.01,
        alpha: float = 0.5,
        pop_neg_sampling: bool = True,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.lambda_ = lambda_
        self.alpha = alpha
        self.pop_neg_sampling = pop_neg_sampling
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None
        self.item_popularity: np.ndarray | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        """Train on the user-item matrix using BPR-SGD with diversity adjustments."""
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

        # Item popularity = number of users who have interacted with each item
        item_counts = np.array(urm.sum(axis=0)).flatten()  # shape (n_items,)
        self.item_popularity = item_counts

        # Sampling weights for popularity-biased negative sampling.
        # Probabilities are proportional to interaction count so that
        # popular items appear as negatives more often during training.
        pop_weights: np.ndarray | None = None
        if self.pop_neg_sampling:
            pop_weights = np.maximum(item_counts.astype(float), 1.0)

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
                    # Build negative candidate set (items not rated by u)
                    neg_mask = np.ones(n_items, dtype=bool)
                    neg_mask[pos_items] = False
                    neg_candidates = all_items[neg_mask]

                    if self.pop_neg_sampling and pop_weights is not None:
                        # Weight negatives by item popularity so popular items
                        # are sampled as negatives proportionally more often
                        neg_weights = pop_weights[neg_candidates]
                        neg_probs = neg_weights / neg_weights.sum()
                        j = int(rng.choice(neg_candidates, p=neg_probs))
                    else:
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
        """Return the latent-factor dot-product score for a user-item pair.

        The popularity penalty is intentionally *not* applied here so that
        RMSE/MAE evaluation against explicit ratings remains valid.
        """
        return float(self.user_factors[user_id] @ self.item_factors[item_id])

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the top-n unrated items ranked by popularity-penalised score.

        The adjusted score for item i is:

            score(u, i) - alpha * log(1 + popularity(i))

        where popularity(i) is the number of interactions item i received
        in the training URM.  Setting alpha=0 recovers the vanilla BPR ranking.
        """
        raw_scores = self.user_factors[user_id] @ self.item_factors.T

        if self.alpha > 0.0 and self.item_popularity is not None:
            penalty = self.alpha * np.log1p(self.item_popularity)
            adjusted_scores = raw_scores - penalty
        else:
            adjusted_scores = raw_scores

        rated_items = set(self.urm[user_id].indices) if self.urm is not None else set()
        candidate_idx = [i for i in range(len(adjusted_scores)) if i not in rated_items]
        top_idx = sorted(candidate_idx, key=lambda i: adjusted_scores[i], reverse=True)[:n]
        return [(i, float(adjusted_scores[i])) for i in top_idx]

    def __str__(self) -> str:
        return "Adjusted BPR (Popularity-Debiased)"

    def __repr__(self) -> str:
        return (
            "AdjustedBPRFactorization("
            f"n_factors={self.n_factors}, "
            f"n_epochs={self.n_epochs}, "
            f"lr={self.lr}, "
            f"lambda_={self.lambda_}, "
            f"alpha={self.alpha}, "
            f"pop_neg_sampling={self.pop_neg_sampling}"
            ")"
        )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    out = path.Path(OUT_DIR)

    urm = sp.load_npz(out / "user_item_matrix.npz")
    val_df = pd.read_csv(out / "val_ratings.csv")
    test_df = pd.read_csv(out / "test_ratings.csv")

    print(f"URM shape: {urm.shape}")

    model = AdjustedBPRFactorization(
        n_factors=50, n_epochs=20, lr=0.05, lambda_=0.01,
        alpha=0.5, pop_neg_sampling=True,
    )
    model.fit(urm)

    # Persist loss history
    loss_path = out / "adjusted_bpr_loss_history.json"
    with open(loss_path, "w") as f:
        json.dump(model.loss_history, f)
    print(f"Saved adjusted BPR loss history to {loss_path}")

    print(f"\nValidation set — ranking metrics (k=10):")
    val_metrics = evaluate_recommendations(model, val_df, k=10)
    print(f"  Precision@10 : {val_metrics['precision_at_k']:.4f}")
    print(f"  Recall@10    : {val_metrics['recall_at_k']:.4f}")
    print(f"  NDCG@10      : {val_metrics['ndcg_at_k']:.4f}")
    print(f"  HitRate@10   : {val_metrics['hit_rate_at_k']:.4f}")


if __name__ == "__main__":
    main()
