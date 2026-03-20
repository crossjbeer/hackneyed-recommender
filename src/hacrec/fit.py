"""Train and checkpoint recommender models using registry defaults."""

import pathlib as path

import scipy.sparse as sp

from .load import load_user_item_matrix
from .recommender import Recommender
from .recommender_registry import registry
from .transform import OUT_DIR


def default_checkpoint_dir() -> path.Path:
    """Return the default location for persisted model checkpoints."""
    return path.Path(OUT_DIR) / "models"


def fit_recommender(
    name: str,
    urm: sp.csr_matrix,
    checkpoint_dir: str | path.Path | None = None,
    force_refit: bool = False,
    **overrides,
) -> Recommender:
    """Build or load a fitted recommender, optionally overriding defaults."""
    return registry.build_or_load(
        name,
        urm,
        checkpoint_dir or default_checkpoint_dir(),
        force_refit=force_refit,
        **overrides,
    )


def fit_recommenders(
    strategies: list[str] | None = None,
    urm: sp.csr_matrix | None = None,
    checkpoint_dir: str | path.Path | None = None,
    force_refit: bool = False,
    strategy_overrides: dict[str, dict] | None = None,
) -> dict[str, Recommender]:
    """Fit and checkpoint the selected recommenders using registry defaults.

    Pass strategy_overrides as a mapping from recommender name to constructor
    keyword overrides when you want to deviate from registry defaults.
    """
    if urm is None:
        raise ValueError("fit_recommenders() requires a user-item matrix via the 'urm' argument")

    selected = strategies or registry.names
    fitted_models: dict[str, Recommender] = {}
    overrides_by_strategy = strategy_overrides or {}

    for name in selected:
        print(f"Fitting Recommender: {name}")
        fitted_models[name] = fit_recommender(
            name,
            urm,
            checkpoint_dir=checkpoint_dir,
            force_refit=force_refit,
            **overrides_by_strategy.get(name, {}),
        )

    return fitted_models


def main() -> None:
    urm = load_user_item_matrix(OUT_DIR)
    fit_recommenders(urm=urm)


if __name__ == "__main__":
    main()