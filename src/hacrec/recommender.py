"""Base recommender interface for recommendation models."""

from abc import ABC, abstractmethod

import scipy.sparse as sp


class Recommender(ABC):
    """Base class for recommendation models.

    All recommenders must implement the following methods:
        - fit(urm): Train the model on the user-item interaction matrix.
        - recommend(user_id, n): Return the top-n recommended item indices with predicted scores for a user.
        - predict(user_id, item_id): Predict a rating for a given user-item pair. Not necessarily supported by all recommendation models.
    """
    @abstractmethod
    def __str__(self) -> str:
        """Return the model name."""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Return a description of the model."""
        ...


    @abstractmethod
    def fit(self, urm: sp.csr_matrix) -> None:
        """Train the model on the user-item interaction matrix."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the top-n recommended item indices with predicted scores for a user."""
        ...

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict a rating for a given user-item pair."""
        ...


def evaluate_predictions(model, eval_df) -> dict:
    """Compatibility wrapper for the centralized evaluation module."""
    from .eval import evaluate_predictions as eval_predictions

    return eval_predictions(model, eval_df)


def evaluate_recommendations(model, eval_df, k: int = 10, relevance_threshold: float = 4.0) -> dict:
    """Compatibility wrapper for the centralized evaluation module."""
    from .eval import evaluate_recommendations as eval_recommendations_impl

    return eval_recommendations_impl(model, eval_df, k=k, relevance_threshold=relevance_threshold)
