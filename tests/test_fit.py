import scipy.sparse as sp
import pytest

from hacrec import fit
from hacrec.recommender import Recommender
from hacrec.recommender_registry import RecommenderRegistry


class DummyRecommender(Recommender):
    def __init__(self, bias: float = 1.0) -> None:
        self.bias = bias
        self.fitted = False
        self.loss_history: list[float] = []

    def fit(self, urm: sp.csr_matrix) -> None:
        self.fitted = True
        self.loss_history = [float(urm.nnz), self.bias]

    def predict(self, user_id: int, item_id: int) -> float:
        return self.bias

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        return [(idx, self.bias) for idx in range(n)]

    def __str__(self) -> str:
        return "DummyRecommender"

    def __repr__(self) -> str:
        return f"DummyRecommender(bias={self.bias})"


def test_fit_recommender_checkpoints_model(tmp_path, monkeypatch):
    local_registry = RecommenderRegistry()
    local_registry.register("dummy", DummyRecommender, {"bias": 2.5})
    monkeypatch.setattr(fit, "registry", local_registry)

    urm = sp.csr_matrix([[1.0, 0.0], [0.0, 3.0]])
    checkpoint_dir = tmp_path / "models"

    model = fit.fit_recommender(
        "dummy",
        urm,
        checkpoint_dir=checkpoint_dir,
        force_refit=True,
    )

    checkpoint_path = local_registry.checkpoint_path("dummy", checkpoint_dir)

    assert model.fitted is True
    assert checkpoint_path.exists()


def test_fit_recommenders_uses_registry_names_by_default(tmp_path, monkeypatch):
    local_registry = RecommenderRegistry()
    local_registry.register("first", DummyRecommender, {"bias": 1.0})
    local_registry.register("second", DummyRecommender, {"bias": 3.0})
    monkeypatch.setattr(fit, "registry", local_registry)

    models = fit.fit_recommenders(
        urm=sp.csr_matrix([[1.0]]),
        checkpoint_dir=tmp_path / "models",
        force_refit=True,
    )

    assert set(models) == {"first", "second"}
    assert all(model.fitted for model in models.values())


def test_fit_recommender_supports_direct_overrides(tmp_path, monkeypatch):
    local_registry = RecommenderRegistry()
    local_registry.register("dummy", DummyRecommender, {"bias": 2.5})
    monkeypatch.setattr(fit, "registry", local_registry)

    model = fit.fit_recommender(
        "dummy",
        sp.csr_matrix([[1.0]]),
        checkpoint_dir=tmp_path / "models",
        force_refit=True,
        bias=7.5,
    )

    checkpoint_path = local_registry.checkpoint_path("dummy", tmp_path / "models", bias=7.5)

    assert model.bias == 7.5
    assert checkpoint_path.exists()


def test_fit_recommenders_supports_strategy_overrides(tmp_path, monkeypatch):
    local_registry = RecommenderRegistry()
    local_registry.register("first", DummyRecommender, {"bias": 1.0})
    local_registry.register("second", DummyRecommender, {"bias": 3.0})
    monkeypatch.setattr(fit, "registry", local_registry)

    models = fit.fit_recommenders(
        strategies=["first", "second"],
        urm=sp.csr_matrix([[1.0]]),
        checkpoint_dir=tmp_path / "models",
        force_refit=True,
        strategy_overrides={"second": {"bias": 9.0}},
    )

    assert models["first"].bias == 1.0
    assert models["second"].bias == 9.0


def test_fit_recommenders_requires_urm(monkeypatch):
    local_registry = RecommenderRegistry()
    local_registry.register("dummy", DummyRecommender, {"bias": 1.0})
    monkeypatch.setattr(fit, "registry", local_registry)

    with pytest.raises(ValueError, match="requires a user-item matrix"):
        fit.fit_recommenders()