"""Central registry for recommender strategies."""

import pathlib

import joblib
import scipy.sparse as sp

from .recommender import Recommender
from .itembasedcf import ItemBasedCF
from .biaseditembasedcf import BiasedCollaborativeCF
from .baselines import (
    GlobalMeanBaseline,
    UserMeanBaseline,
    ItemMeanBaseline,
    UserItemBiasBaseline,
    MostPopularBaseline,
    RandomRecommender,
)
from .alsfactorization import ALSFactorization
from .biasedalsfactorization import BiasedALSFactorization
from .bprfactorization import BPRFactorization
from .adjustedbprfactorization import AdjustedBPRFactorization
from .implicitalsfactorization import ImplicitALSFactorizer


class RecommenderRegistry:
    """Registry mapping strategy names to their classes and default parameters."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[type[Recommender], dict]] = {}

    def register(
        self,
        name: str,
        cls: type[Recommender],
        default_params: dict | None = None,
    ) -> None:
        """Register a strategy under *name*."""
        self._entries[name] = (cls, default_params or {})

    def build(self, name: str, **overrides) -> Recommender:
        """Instantiate the named strategy (unfitted), optionally overriding defaults."""
        if name not in self._entries:
            raise ValueError(
                f"Unknown strategy '{name}'. Available: {self.names}"
            )
        cls, defaults = self._entries[name]
        return cls(**{**defaults, **overrides})

    def build_or_load(
        self,
        name: str,
        urm: sp.csr_matrix,
        checkpoint_dir: str | pathlib.Path,
        force_refit: bool = False,
        **overrides,
    ) -> Recommender:
        """Return a fitted model, using a checkpoint when available.

        If a checkpoint for *name* already exists in *checkpoint_dir* and
        *force_refit* is False, the model is loaded directly.  Otherwise the
        model is fitted from scratch and the checkpoint is written.
        """
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{name}.pkl"

        if not force_refit and checkpoint_path.exists():
            return joblib.load(checkpoint_path)

        model = self.build(name, **overrides)
        model.fit(urm)
        joblib.dump(model, checkpoint_path)
        return model

    @property
    def names(self) -> list[str]:
        return list(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __repr__(self) -> str:
        return f"RecommenderRegistry({self.names})"


# --------------------------------------------------------------------------
# Default registry — all built-in strategies
# --------------------------------------------------------------------------

registry = RecommenderRegistry()

registry.register("item-based-cf",  ItemBasedCF,             {"k": 50})
registry.register("als",            ALSFactorization,         {"n_factors": 65, "n_iterations": 40, "lambda_": 0.1})
registry.register("global-mean",    GlobalMeanBaseline,       {})
registry.register("user-mean",      UserMeanBaseline,         {})
registry.register("item-mean",      ItemMeanBaseline,         {})
registry.register("user-item-bias", UserItemBiasBaseline,     {"reg": 10.0, "n_iterations": 10})
registry.register("most-popular",   MostPopularBaseline,      {})
registry.register("random",         RandomRecommender,        {"seed": 42})
registry.register("biased-als",     BiasedALSFactorization,   {"n_factors": 20, "n_iterations": 40, "lambda_": 0.1})
registry.register("biased-item-cf", BiasedCollaborativeCF,    {"k": 50, "reg": 10.0})
registry.register("bpr",            BPRFactorization,         {"n_factors": 50, "n_epochs": 20, "lr": 0.05, "lambda_": 0.01})
registry.register("adjusted-bpr",   AdjustedBPRFactorization, {"n_factors": 50, "n_epochs": 20, "lr": 0.05, "lambda_": 0.01})
registry.register("implicit-als",   ImplicitALSFactorizer,    {"n_factors": 50, "n_iterations": 15, "lambda_": 0.1, "alpha": 40.0})
