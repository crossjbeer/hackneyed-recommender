"""Central registry for recommender strategies."""

import csv
import datetime
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

    # ------------------------------------------------------------------
    # models.csv helpers
    # ------------------------------------------------------------------

    def _csv_path(self, checkpoint_dir: pathlib.Path) -> pathlib.Path:
        return checkpoint_dir / "models.csv"

    def _append_csv(
        self,
        checkpoint_dir: pathlib.Path,
        name: str,
        dt: str,
        params: dict,
    ) -> None:
        csv_path = self._csv_path(checkpoint_dir)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "datetime", "params"])
            if write_header:
                writer.writeheader()
            writer.writerow({"name": name, "datetime": dt, "params": repr(params)})

    def _find_in_csv(
        self,
        checkpoint_dir: pathlib.Path,
        name: str,
        params: dict,
    ) -> pathlib.Path | None:
        """Return the pkl path of an existing checkpoint matching name+params, or None."""
        csv_path = self._csv_path(checkpoint_dir)
        if not csv_path.exists():
            return None
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["name"] != name:
                    continue
                try:
                    saved_params = eval(row["params"])  # noqa: S307
                except Exception:
                    continue
                if saved_params == params:
                    pkl = checkpoint_dir / f"{name}_{row['datetime']}" / "model.pkl"
                    if pkl.exists():
                        return pkl
        return None

    # ------------------------------------------------------------------
    # Core fit / load
    # ------------------------------------------------------------------

    def build_or_load(
        self,
        name: str,
        urm: sp.csr_matrix,
        checkpoint_dir: str | pathlib.Path,
        force_refit: bool = False,
        **overrides,
    ) -> Recommender:
        """Return a fitted model, loading from an existing checkpoint when available.

        Each new fit is saved to ``{checkpoint_dir}/{name}_{YYMMDD_HHMMSS}/model.pkl``
        and a row is appended to ``{checkpoint_dir}/models.csv``.  When
        *force_refit* is False and a checkpoint with matching name + params
        already exists in the CSV, that model is loaded instead.
        """
        if name not in self._entries:
            raise ValueError(
                f"Unknown strategy '{name}'. Available: {self.names}"
            )

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        _, defaults = self._entries[name]
        effective_params = {**defaults, **overrides}

        if not force_refit:
            existing = self._find_in_csv(checkpoint_dir, name, effective_params)
            if existing:
                return joblib.load(existing)

        model = self.build(name, **overrides)
        model.fit(urm)

        dt = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        folder = checkpoint_dir / f"{name}_{dt}"
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / "model.pkl")
        self._append_csv(checkpoint_dir, name, dt, effective_params)

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
