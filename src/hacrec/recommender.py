"""Base recommender interface for recommendation models."""

from abc import ABC, abstractmethod

import scipy.sparse as sp


class Recommender(ABC):
    """Base class for recommendation models.

    Subclasses must implement fit() and predict() so that
    evaluate_predictions() can work with any model uniformly.
    """

    @abstractmethod
    def fit(self, urm: sp.csr_matrix) -> None:
        """Train the model on the user-item interaction matrix."""
        ...

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict a rating for a given user-item pair."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the top-n recommended item indices with predicted scores for a user."""
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Return the model name."""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Return a description of the model."""
        ...


def evaluate_predictions(model, eval_df) -> dict:
    """Compatibility wrapper for the centralized evaluation module."""
    from .eval import evaluate_predictions as eval_predictions

    return eval_predictions(model, eval_df)


def evaluate_recommendations(model, eval_df, k: int = 10, relevance_threshold: float = 4.0) -> dict:
    """Compatibility wrapper for the centralized evaluation module."""
    from .eval import evaluate_recommendations as eval_recommendations_impl

    return eval_recommendations_impl(model, eval_df, k=k, relevance_threshold=relevance_threshold)
