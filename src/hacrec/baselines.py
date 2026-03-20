"""Naive recommendation baseline strategies."""

import numpy as np
import scipy.sparse as sp

from .recommender import Recommender


class GlobalMeanBaseline(Recommender):
    """Naive baseline: predict the global mean rating for all pairs."""

    def __init__(self):
        self.global_mean: float = 0.0
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm
        self.global_mean = float(np.mean(urm.data)) if urm.nnz > 0 else 0.0

    def predict(self, user_id: int, item_id: int) -> float:
        return self.global_mean

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        num_items = self.urm.shape[1]
        rated_items = set(self.urm[user_id].indices)
        scores = [
            (item_id, self.global_mean)
            for item_id in range(num_items)
            if item_id not in rated_items
        ]
        return scores[:n]

    def __str__(self) -> str:
        return "Global Mean Baseline"

    def __repr__(self) -> str:
        return "GlobalMeanBaseline()"


class UserMeanBaseline(Recommender):
    """Naive baseline: predict each user's average rating."""

    def __init__(self):
        self.global_mean: float = 0.0
        self.user_means: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm
        self.global_mean = float(np.mean(urm.data)) if urm.nnz > 0 else 0.0

        sums = np.asarray(urm.sum(axis=1)).ravel()
        counts = np.diff(urm.indptr)
        means = np.full(urm.shape[0], self.global_mean, dtype=float)
        nonzero = counts > 0
        means[nonzero] = sums[nonzero] / counts[nonzero]
        self.user_means = means

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id >= len(self.user_means):
            return self.global_mean
        return float(self.user_means[user_id])

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        score = self.predict(user_id=user_id, item_id=0)
        num_items = self.urm.shape[1]
        rated_items = set(self.urm[user_id].indices) if user_id < self.urm.shape[0] else set()
        scores = [
            (item_id, score)
            for item_id in range(num_items)
            if item_id not in rated_items
        ]
        return scores[:n]

    def __str__(self) -> str:
        return "User Mean Baseline"

    def __repr__(self) -> str:
        return "UserMeanBaseline()"


class ItemMeanBaseline(Recommender):
    """Naive baseline: predict each item's average rating."""

    def __init__(self):
        self.global_mean: float = 0.0
        self.item_means: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm
        self.global_mean = float(np.mean(urm.data)) if urm.nnz > 0 else 0.0

        urm_csc = urm.tocsc()
        sums = np.asarray(urm_csc.sum(axis=0)).ravel()
        counts = np.diff(urm_csc.indptr)
        means = np.full(urm.shape[1], self.global_mean, dtype=float)
        nonzero = counts > 0
        means[nonzero] = sums[nonzero] / counts[nonzero]
        self.item_means = means

    def predict(self, user_id: int, item_id: int) -> float:
        if item_id >= len(self.item_means):
            return self.global_mean
        return float(self.item_means[item_id])

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        rated_items = set(self.urm[user_id].indices) if user_id < self.urm.shape[0] else set()
        candidate_idx = [i for i in range(self.urm.shape[1]) if i not in rated_items]
        top_idx = sorted(candidate_idx, key=lambda i: self.item_means[i], reverse=True)[:n]
        return [(i, float(self.item_means[i])) for i in top_idx]

    def __str__(self) -> str:
        return "Item Mean Baseline"

    def __repr__(self) -> str:
        return "ItemMeanBaseline()"


class RandomRecommender(Recommender):
    """Sanity-check baseline: recommend uniformly random unrated items."""

    def __init__(self, seed: int | None = None):
        self.urm: sp.csr_matrix | None = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm

    def predict(self, user_id: int, item_id: int) -> float:
        return float(self.rng.random())

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        num_items = self.urm.shape[1]
        rated_items = set(self.urm[user_id].indices) if user_id < self.urm.shape[0] else set()
        candidates = [i for i in range(num_items) if i not in rated_items]
        chosen = self.rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(int(i), float(self.rng.random())) for i in chosen]

    def __str__(self) -> str:
        return "Random (Sanity Check)"

    def __repr__(self) -> str:
        return f"RandomRecommender(seed={self.seed})"


class UserItemBiasBaseline(Recommender):
    """Bias-only baseline: global mean + user bias + item bias."""

    def __init__(self, reg: float = 10.0, n_iterations: int = 10):
        self.reg = reg
        self.n_iterations = n_iterations
        self.global_mean: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm
        n_users, n_items = urm.shape
        self.global_mean = float(np.mean(urm.data)) if urm.nnz > 0 else 0.0
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

    def predict(self, user_id: int, item_id: int) -> float:
        bu = self.user_bias[user_id] if user_id < len(self.user_bias) else 0.0
        bi = self.item_bias[item_id] if item_id < len(self.item_bias) else 0.0
        return float(self.global_mean + bu + bi)

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        bu = self.user_bias[user_id] if user_id < len(self.user_bias) else 0.0
        scores = self.global_mean + bu + self.item_bias
        rated_items = set(self.urm[user_id].indices) if user_id < self.urm.shape[0] else set()
        candidate_idx = [i for i in range(len(scores)) if i not in rated_items]
        top_idx = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:n]
        return [(i, float(scores[i])) for i in top_idx]

    def __str__(self) -> str:
        return "User-Item Bias Baseline"

    def __repr__(self) -> str:
        return (
            f"UserItemBiasBaseline(reg={self.reg}, n_iterations={self.n_iterations})"
        )


class MostPopularBaseline(Recommender):
    """Naive ranking baseline: recommend most frequently rated items."""

    def __init__(self):
        self.global_mean: float = 0.0
        self.item_counts: np.ndarray | None = None
        self.item_means: np.ndarray | None = None
        self.urm: sp.csr_matrix | None = None

    def fit(self, urm: sp.csr_matrix) -> None:
        self.urm = urm
        self.global_mean = float(np.mean(urm.data)) if urm.nnz > 0 else 0.0

        urm_csc = urm.tocsc()
        sums = np.asarray(urm_csc.sum(axis=0)).ravel()
        counts = np.diff(urm_csc.indptr)
        means = np.full(urm.shape[1], self.global_mean, dtype=float)
        nonzero = counts > 0
        means[nonzero] = sums[nonzero] / counts[nonzero]

        self.item_counts = counts.astype(float)
        self.item_means = means

    def predict(self, user_id: int, item_id: int) -> float:
        if item_id >= len(self.item_means):
            return self.global_mean
        return float(self.item_means[item_id])

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        rated_items = set(self.urm[user_id].indices) if user_id < self.urm.shape[0] else set()
        candidate_idx = [i for i in range(self.urm.shape[1]) if i not in rated_items]
        top_idx = sorted(
            candidate_idx,
            key=lambda i: (self.item_counts[i], self.item_means[i]),
            reverse=True,
        )[:n]
        return [(i, float(self.item_means[i])) for i in top_idx]

    def __str__(self) -> str:
        return "Most Popular Baseline"

    def __repr__(self) -> str:
        return "MostPopularBaseline()"
