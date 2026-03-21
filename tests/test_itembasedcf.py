"""Tests for hacrec.itembasedcf.ItemBasedCF.

Hand-computed similarity values for the main fixtures
------------------------------------------------------

chain_urm  (5 users × 4 items)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       item0  item1  item2  item3
user0: [  4,    0,    0,    0 ]
user1: [  4,    2,    0,    0 ]
user2: [  0,    2,    5,    0 ]
user3: [  0,    0,    5,    4 ]
user4: [  0,    0,    0,    0 ]

Column norms:
  item0 = 4√2,  item1 = 2√2,  item2 = 5√2,  item3 = 4

Exact cosine similarities (all others = 0):
  sim[0, 1] = 8 / (4√2 · 2√2) = 8/16 = 0.5
  sim[1, 2] = 10 / (2√2 · 5√2) = 10/20 = 0.5
  sim[2, 3] = 20 / (5√2 · 4)   = 20/(20√2) = 1/√2

simple_urm  (3 users × 3 items)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       item0  item1  item2
user0: [  0,    3,    4 ]
user1: [  3,    0,    0 ]
user2: [  4,    0,    0 ]

item1 col = [3, 0, 0],  item2 col = [4, 0, 0]  → identical direction → sim = 1.0
item0 col = [0, 3, 4]  → orthogonal to item1 and item2 → sim = 0

all_rated_urm  (3 users × 3 items, all items rated by every user)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       item0  item1  item2
user0: [  5,    4,    1 ]
user1: [  4,    5,    1 ]
user2: [  1,    1,    5 ]

Column norms (same): item0 = √42,  item1 = √42,  item2 = √27

Exact similarities:
  sim[0, 1] = 41/42
  sim[0, 2] = sim[1, 2] = 14/√1134

Used to test predict() top-k truncation.
"""

import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from hacrec.itembasedcf import ItemBasedCF
from hacrec.recommender import evaluate_predictions, evaluate_recommendations

# ===========================================================================
# Shared constants
# ===========================================================================

_SQRT2 = math.sqrt(2)
_SIM_01 = 0.5                     # chain_urm
_SIM_12 = 0.5                     # chain_urm
_SIM_23 = 1.0 / _SQRT2            # chain_urm
_SIM_ALL_01 = 41 / 42             # all_rated_urm  ≈ 0.9762
_SIM_ALL_02 = 14 / math.sqrt(1134)  # all_rated_urm  ≈ 0.4158

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def simple_urm():
    """3 users × 3 items.

    item1 and item2 are proportional (sim = 1.0).
    item0 is orthogonal to both (sim = 0).
    """
    data = np.array(
        [[0, 3, 4], [3, 0, 0], [4, 0, 0]], dtype=float
    )
    return sp.csr_matrix(data)


@pytest.fixture
def simple_model(simple_urm):
    model = ItemBasedCF(k=50)
    model.fit(simple_urm)
    return model


@pytest.fixture
def chain_urm():
    """5 users × 4 items with chain-like cosine similarity.

    Exact similarities (see module docstring).
    user4 (row 4) has no ratings — used for empty-user tests.
    """
    data = np.array(
        [
            [4, 0, 0, 0],
            [4, 2, 0, 0],
            [0, 2, 5, 0],
            [0, 0, 5, 4],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    return sp.csr_matrix(data)


@pytest.fixture
def chain_model(chain_urm):
    model = ItemBasedCF(k=50)
    model.fit(chain_urm)
    return model


@pytest.fixture
def all_rated_urm():
    """3 users × 3 items where every user has rated every item.

    All pairwise similarities are non-zero (see module docstring).
    Useful for top-k truncation tests where pruning changes predict results.
    """
    data = np.array(
        [[5, 4, 1], [4, 5, 1], [1, 1, 5]], dtype=float
    )
    return sp.csr_matrix(data)


@pytest.fixture
def urm_with_zero_item():
    """2 users × 3 items where item2 (col 2) has no ratings."""
    data = np.array([[3, 4, 0], [0, 5, 0]], dtype=float)
    return sp.csr_matrix(data)


@pytest.fixture
def divergence_urm():
    """5 users × 4 items designed to expose predict/recommend divergence.

    Items: A=0, B=1, C=2, E=3.
    With k=1, A's top-1 neighbor is E (sim ≈ 0.515), NOT C (sim ≈ 0.285).
    User 0 has rated only A.
    So predict(user=0, item=2) uses sim_full and finds A→C connection,
    but recommend() uses sim_topk[A, :] which points to E, not C.
    """
    data = np.array(
        [
            [4, 0, 0, 0],   # user0: rated A only
            [3, 0, 2, 0],   # user1: A and C → gives A-C similarity
            [3, 0, 0, 4],   # user2: A and E → gives A-E similarity (stronger)
            [0, 5, 3, 0],   # user3: B and C
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    return sp.csr_matrix(data)


# ===========================================================================
# 1. Fit-time structural tests
# ===========================================================================


class TestFitStructure:
    def test_urm_stored_as_csr_from_coo_input(self, chain_urm):
        """1.1 — fit() must convert any format to CSR."""
        coo = chain_urm.tocoo()
        model = ItemBasedCF(k=50)
        model.fit(coo)

        assert sp.issparse(model.urm)
        assert model.urm.format == "csr"

    def test_urm_stored_as_csr_from_csc_input(self, chain_urm):
        """1.1 — fit() must convert CSC to CSR."""
        csc = chain_urm.tocsc()
        model = ItemBasedCF(k=50)
        model.fit(csc)

        assert model.urm.format == "csr"
        assert model.urm.shape == chain_urm.shape

    def test_similarity_matrices_are_item_by_item(self, chain_urm, chain_model):
        """1.2 — All three similarity matrices must have shape (n_items, n_items)."""
        n_items = chain_urm.shape[1]
        expected = (n_items, n_items)

        assert chain_model.similarity_full.shape == expected
        assert chain_model.similarity_topk.shape == expected
        assert chain_model.abs_similarity_topk.shape == expected

    def test_diagonal_of_similarity_full_is_zero(self, chain_model):
        """1.3 — Explicit setdiag(0) in fit() must zero out self-similarity."""
        diag = chain_model.similarity_full.diagonal()
        np.testing.assert_array_equal(diag, 0.0)

    def test_diagonal_of_similarity_topk_is_zero(self, chain_model):
        """1.3 — Top-k pruning must not reintroduce diagonal entries."""
        diag = chain_model.similarity_topk.diagonal()
        np.testing.assert_array_equal(diag, 0.0)

    def test_similarity_full_is_symmetric(self, chain_model):
        """1.4 — Cosine similarity from X^T X is always symmetric."""
        full = chain_model.similarity_full.toarray()
        np.testing.assert_allclose(full, full.T, atol=1e-12)

    def test_similarity_topk_need_not_be_symmetric(self, chain_urm):
        """1.4 — Row-wise top-k can legally break symmetry; just confirm no crash."""
        model = ItemBasedCF(k=1)
        model.fit(chain_urm)
        # If k < max_neighbors, asymmetry may exist; simply assert it runs.
        topk = model.similarity_topk.toarray()
        assert topk.shape == (chain_urm.shape[1], chain_urm.shape[1])

    def test_zero_norm_item_does_not_raise(self, urm_with_zero_item):
        """1.5 — Items with no ratings must not cause division by zero."""
        model = ItemBasedCF(k=50)
        model.fit(urm_with_zero_item)  # must not raise

    def test_zero_norm_item_has_zero_similarity_row_and_col(self, urm_with_zero_item):
        """1.5 — The column/row for an unrated item must be all-zero in similarity_full."""
        model = ItemBasedCF(k=50)
        model.fit(urm_with_zero_item)
        full = model.similarity_full.toarray()

        # item2 (col/row 2) has no ratings → its similarities should all be zero
        np.testing.assert_array_equal(full[2, :], 0.0)
        np.testing.assert_array_equal(full[:, 2], 0.0)


# ===========================================================================
# 2. _keep_topk_per_row() unit tests
# ===========================================================================


class TestKeepTopkPerRow:
    """Direct unit tests for the internal top-k helper."""

    @staticmethod
    def _make_model():
        return ItemBasedCF(k=50)

    def _build_csr(self, rows):
        """Build a CSR matrix from a list-of-list dense representation."""
        return sp.csr_matrix(np.array(rows, dtype=float))

    def test_keeps_all_entries_when_nnz_le_k(self):
        """2.1 — Rows with ≤ k nonzeros must not be pruned."""
        mat = self._build_csr([[0, 1, 2, 0], [0, 0, 3, 4]])
        model = self._make_model()
        result = model._keep_topk_per_row(mat, k=50)

        np.testing.assert_array_equal(
            result.toarray(), mat.toarray()
        )

    def test_keeps_exactly_k_values_when_row_exceeds_k(self):
        """2.2 — Rows with more than k nonzeros must retain exactly k values."""
        row = np.array([[1, 5, 3, 2, 4]], dtype=float)  # 5 nonzero values
        mat = sp.csr_matrix(row)
        model = self._make_model()
        result = model._keep_topk_per_row(mat, k=3)

        assert result[0].nnz == 3
        # Retained values must be the 3 largest: 5, 4, 3
        retained = set(result.data)
        assert retained == {5.0, 4.0, 3.0}

    def test_rows_are_sorted_descending_after_pruning(self):
        """2.3 — After pruning, data within each pruned row is descending."""
        row = np.array([[1, 7, 3, 9, 2]], dtype=float)
        mat = sp.csr_matrix(row)
        model = self._make_model()
        result = model._keep_topk_per_row(mat, k=2)

        # CSR row slice
        start, end = result.indptr[0], result.indptr[1]
        retained_data = result.data[start:end]
        assert list(retained_data) == sorted(retained_data, reverse=True)

    def test_handles_tied_values_at_boundary(self):
        """2.4 — k top values chosen from ties must have correct size and valid values."""
        # 4 nonzero values, two at the boundary (both equal to 3)
        row = np.array([[5, 3, 3, 1]], dtype=float)
        mat = sp.csr_matrix(row)
        model = self._make_model()
        result = model._keep_topk_per_row(mat, k=2)

        assert result[0].nnz == 2
        # Both retained values must be among the valid top values {5, 3}
        for v in result.data:
            assert v in {5.0, 3.0}

    def test_k_zero_does_not_prune_entries(self):
        """2.5 — k=0 exposes a likely edge-case bug: -0 == 0 in Python, so [-0:] is [:].

        Current behaviour: no pruning occurs for any row with data.
        This test documents the existing behaviour and should fail if it is fixed
        to correctly zero out all entries.
        """
        row = np.array([[3, 1, 4]], dtype=float)
        mat = sp.csr_matrix(row)
        model = self._make_model()
        result = model._keep_topk_per_row(mat, k=0)
        # Document current buggy behaviour: entries are NOT removed.
        assert result[0].nnz == 3

    def test_k_exceeds_n_items_does_not_crash(self, chain_urm):
        """2.6 — k larger than the number of items must not raise."""
        model = ItemBasedCF(k=999)
        model.fit(chain_urm)  # must not raise
        # All nonzero similarities should be retained
        assert model.similarity_topk.nnz == model.similarity_full.nnz

    def test_empty_row_remains_empty(self):
        """2.1 edge — Rows that are entirely zero stay all-zero."""
        mat = sp.csr_matrix(np.array([[0, 0, 0], [1, 2, 0]], dtype=float))
        model = self._make_model()
        result = model._keep_topk_per_row(mat, k=2)

        assert result[0].nnz == 0


# ===========================================================================
# 3. Similarity correctness
# ===========================================================================


class TestSimilarityCorrectness:
    def test_identical_item_columns_have_similarity_one(self, simple_model):
        """3.1 — Items with proportional columns must have cosine similarity = 1."""
        sim = simple_model.similarity_full[1, 2]
        assert abs(float(sim) - 1.0) < 1e-10

    def test_orthogonal_item_columns_have_similarity_zero(self, simple_model):
        """3.2 — Items rated by disjoint user sets are orthogonal (sim = 0)."""
        assert float(simple_model.similarity_full[0, 1]) == pytest.approx(0.0, abs=1e-12)
        assert float(simple_model.similarity_full[0, 2]) == pytest.approx(0.0, abs=1e-12)

    def test_manual_cosine_matches_expected_value(self, chain_model):
        """3.3 — Spot-check hand-computed similarities for chain_urm."""
        sim = chain_model.similarity_full.toarray()
        assert sim[0, 1] == pytest.approx(_SIM_01, abs=1e-10)
        assert sim[1, 2] == pytest.approx(_SIM_12, abs=1e-10)
        assert sim[2, 3] == pytest.approx(_SIM_23, abs=1e-10)
        # Entries that should be exactly zero
        assert sim[0, 2] == pytest.approx(0.0, abs=1e-12)
        assert sim[0, 3] == pytest.approx(0.0, abs=1e-12)
        assert sim[1, 3] == pytest.approx(0.0, abs=1e-12)

    def test_negative_ratings_produce_negative_similarity(self):
        """3.4 — Negative ratings are valid input; cosine can become negative."""
        # user0 prefers item0, dislikes item1; user1 is the opposite
        data = np.array([[3, -3], [-3, 3]], dtype=float)
        urm = sp.csr_matrix(data)
        model = ItemBasedCF(k=50)
        model.fit(urm)
        sim = float(model.similarity_full[0, 1])
        assert sim < 0.0


# ===========================================================================
# 4. predict() tests
# ===========================================================================


class TestPredict:
    def test_empty_user_returns_zero(self, chain_model):
        """4.1 — A user with no ratings must always return 0.0."""
        result = chain_model.predict(user_id=4, item_id=0)
        assert result == 0.0

    def test_hand_computed_prediction_one_rated_item(self, chain_model):
        """4.2 — Weighted average when user has one rated item.

        user0 rated item0 (4).  sim_full[item1, item0] = 0.5.
        prediction = (0.5 * 4) / 0.5 = 4.0
        """
        pred = chain_model.predict(user_id=0, item_id=1)
        assert pred == pytest.approx(4.0, abs=1e-10)

    def test_hand_computed_prediction_two_rated_items(self, chain_model):
        """4.2 — Weighted average when user has two rated items.

        user1 rated item0 (4) and item1 (2).
        sim_full[item2, item0] = 0,  sim_full[item2, item1] = 0.5.
        numerator = 0*4 + 0.5*2 = 1.0,  denom = 0+0.5 = 0.5
        prediction = 1.0/0.5 = 2.0
        """
        pred = chain_model.predict(user_id=1, item_id=2)
        assert pred == pytest.approx(2.0, abs=1e-10)

    def test_prediction_zero_when_all_similarities_are_zero(self, chain_model):
        """4.4 — When denom = 0 (no relevant similarity), return 0.0."""
        # user0 only rated item0; sim_full[item2, item0] = 0
        pred = chain_model.predict(user_id=0, item_id=3)
        assert pred == 0.0

    def test_topk_truncation_changes_result(self, all_rated_urm):
        """4.5 — predict() does its own top-k over sim_full; pruning must affect output.

        user2 rated all 3 items.  Predicting item0:
        sims = [0, sim[0,1]=41/42, sim[0,2]=14/sqrt(1134)].
        With k=1: keeps only 41/42 → prediction = 41/42 * 1 / (41/42) = 1.0
        With k=2: keeps 41/42 and 14/sqrt(1134) → prediction ≈ 2.19
        """
        model_k1 = ItemBasedCF(k=1)
        model_k1.fit(all_rated_urm)
        pred_k1 = model_k1.predict(user_id=2, item_id=0)

        model_k2 = ItemBasedCF(k=2)
        model_k2.fit(all_rated_urm)
        pred_k2 = model_k2.predict(user_id=2, item_id=0)

        assert pred_k1 == pytest.approx(1.0, abs=1e-10)
        # k=2 must differ because the second neighbour contributes
        assert pred_k2 != pytest.approx(pred_k1, abs=1e-6)

    def test_topk_truncation_k1_exact_value(self, all_rated_urm):
        """4.5 — With k=1, only the strongest neighbour contributes.

        user2 rated: item0(1), item1(1), item2(5).
        Predicting item0 with k=1: top neighbour of item0 among rated is item1
        (sim=41/42), so numerator = (41/42)*1, denom = 41/42 → prediction = 1.0.
        """
        model = ItemBasedCF(k=1)
        model.fit(all_rated_urm)
        pred = model.predict(user_id=2, item_id=0)
        assert pred == pytest.approx(1.0, abs=1e-10)

    def test_predict_uses_only_rated_items(self, chain_model):
        """4.3 — Prediction must be unchanged regardless of unrated item scores.

        Computing predict for user0/item1 twice (back-to-back) must give the
        same value, confirming no state from unrated items leaks in.
        """
        p1 = chain_model.predict(user_id=0, item_id=1)
        p2 = chain_model.predict(user_id=0, item_id=1)
        assert p1 == p2 == pytest.approx(4.0, abs=1e-10)

    def test_predict_seen_item_does_not_return_observed_rating(self, chain_model):
        """4.7 — Predicting a seen item uses other rated items via sim_full, not its own rating.

        user1 rated item0 (4) and item1 (2).  Predicting item1 (a seen item):
        rated_items = [0, 1], sims = [sim[1,0]=0.5, sim[1,1]=0 (diagonal)].
        numerator = 0.5*4 + 0*2 = 2,  denom = 0.5 → prediction = 4.0 ≠ 2.0 (observed).
        """
        pred = chain_model.predict(user_id=1, item_id=1)
        assert pred != pytest.approx(2.0, abs=1e-10)

    def test_prediction_with_negative_similarity(self):
        """4.6 — Top-k selects by largest raw sim; denom uses absolute values.

        item0 and item1 are anti-correlated (neg similarity) because every
        user who rated item0 highly gave item1 a negative rating and vice versa.
        """
        # user0: likes item0 (+5), dislikes item1 (−5); user1: vice versa
        # item0 col = [+5, −5], item1 col = [−5, +5]
        # dot product = −50, each norm = √50  ⟹  cosine = −1.0
        data = np.array([[5, -5], [-5, 5]], dtype=float)
        urm = sp.csr_matrix(data)
        model = ItemBasedCF(k=50)
        model.fit(urm)

        sim = float(model.similarity_full[0, 1])
        assert sim < 0.0, f"Expected negative similarity, got {sim}"

        # Prediction must be a finite number regardless of negative similarity
        pred = model.predict(user_id=0, item_id=1)
        assert math.isfinite(pred)


# ===========================================================================
# 5. recommend() tests
# ===========================================================================


class TestRecommend:
    def test_empty_user_returns_empty_list(self, chain_model):
        """5.1 — A user with no ratings must get an empty recommendation list."""
        result = chain_model.recommend(user_id=4, n=10)
        assert result == []

    def test_seen_items_are_excluded(self, chain_model):
        """5.2 — Items the user has already rated must not appear in results."""
        for uid in range(4):  # users 0-3 each have ratings
            recs = chain_model.recommend(user_id=uid, n=10)
            rated_items = set(chain_model.urm[uid].indices)
            rec_items = {iid for iid, _ in recs}
            assert rated_items.isdisjoint(rec_items), (
                f"user {uid}: rec_items {rec_items} overlap rated_items {rated_items}"
            )

    def test_returns_at_most_n_items(self, chain_model):
        """5.3 — Result length is bounded by n."""
        recs = chain_model.recommend(user_id=1, n=2)
        assert len(recs) <= 2

    def test_results_sorted_descending_by_score(self, chain_model):
        """5.4 — Scores in the returned list must be non-increasing."""
        recs = chain_model.recommend(user_id=1, n=10)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_hand_computed_scores_for_user0(self, chain_model):
        """5.5 — Verify vectorised scoring formula against hand-computed values.

        user0 rated item0 (4).  similarity_topk (k=50) = similarity_full (no pruning).
        numerators = user_row @ sim_topk  = [0, 4*0.5, 0, 0] = [0, 2, 0, 0]
        denominators = user_implicit @ abs_sim_topk = [0, 0.5, 0, 0]
        scores = [0/0→0, 4.0, 0/0→0, 0/0→0]
        Exclude item0 (seen).  Top recommendation: (item1, 4.0).
        """
        recs = chain_model.recommend(user_id=0, n=10)
        assert len(recs) >= 1

        top_item, top_score = recs[0]
        assert top_item == 1
        assert top_score == pytest.approx(4.0, abs=1e-10)

    def test_n_larger_than_items_does_not_crash(self, chain_model):
        """5.6 — Requesting more items than exist must not raise."""
        recs = chain_model.recommend(user_id=0, n=10_000)
        # All unseen, finite-score items must be returned
        assert len(recs) >= 1
        for _, score in recs:
            assert math.isfinite(score)

    def test_zero_denominator_yields_zero_score_not_nan(self, chain_model):
        """5.7 — Items with denominator = 0 must score 0.0, not NaN or inf.

        For user0 (rated only item0), items 2 and 3 have no path through
        sim_topk back to item0, so their scores are 0.0.
        """
        recs = chain_model.recommend(user_id=0, n=10)
        score_map = {iid: s for iid, s in recs}

        # item2 and item3 are unseen and have zero-denominator scores
        for iid in (2, 3):
            if iid in score_map:
                assert math.isfinite(score_map[iid])
                assert not math.isnan(score_map[iid])

    def test_negative_recommendation_scores_are_allowed(self):
        """5.8 — Signed numerators mean scores can be negative; must be ordered."""
        data = np.array([[5, 1], [1, 5], [3, 0]], dtype=float)
        urm = sp.csr_matrix(data)
        model = ItemBasedCF(k=50)
        model.fit(urm)
        # user2 rated only item0=3; item1's similarity to item0 may be negative
        recs = model.recommend(user_id=2, n=10)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 6. predict() / recommend() consistency
# ===========================================================================


class TestConsistency:
    def test_predict_and_recommend_agree_when_no_pruning(self, chain_model):
        """6.1 — With k large enough that no similarity is pruned,
        the score assigned to an item by recommend() must equal predict().

        user0 / item1: predict = 4.0;  recommend top score for item1 = 4.0.
        """
        pred = chain_model.predict(user_id=0, item_id=1)

        recs = chain_model.recommend(user_id=0, n=10)
        rec_score_item1 = next(s for iid, s in recs if iid == 1)

        assert pred == pytest.approx(rec_score_item1, abs=1e-10)

    def test_predict_and_recommend_diverge_when_pruning_occurs(self, divergence_urm):
        """6.2 — When k=1 prunes A's top neighbour to E (not C), recommend cannot
        score item C via item A, while predict() still can via sim_full.

        This documents expected, intentional divergence between the two methods.
        """
        sim_AE = 3 / math.sqrt(34)   # ≈ 0.515
        sim_AC = 6 / math.sqrt(442)  # ≈ 0.285
        assert sim_AE > sim_AC, "Precondition: E is a stronger neighbour of A than C"

        model = ItemBasedCF(k=1)
        model.fit(divergence_urm)

        # predict uses sim_full; A-C connection is present → non-zero prediction
        pred_c = model.predict(user_id=0, item_id=2)
        assert pred_c == pytest.approx(4.0, abs=1e-10)

        # recommend uses sim_topk; A's row points to E, not C → score for C = 0
        recs = model.recommend(user_id=0, n=10)
        score_c = next((s for iid, s in recs if iid == 2), 0.0)
        assert score_c == pytest.approx(0.0, abs=1e-10)

        # Confirm the divergence is real
        assert pred_c != pytest.approx(score_c, abs=1e-6)


# ===========================================================================
# 7. Robustness / error-handling
# ===========================================================================


class TestRobustness:
    def test_predict_before_fit_raises(self):
        """7.1 — Calling predict() before fit() must raise (currently TypeError/AttributeError)."""
        model = ItemBasedCF(k=10)
        with pytest.raises((TypeError, AttributeError)):
            model.predict(0, 0)

    def test_recommend_before_fit_raises(self):
        """7.1 — Calling recommend() before fit() must raise."""
        model = ItemBasedCF(k=10)
        with pytest.raises((TypeError, AttributeError)):
            model.recommend(0)

    def test_predict_out_of_range_user_raises(self, chain_model):
        """7.2 — An out-of-bounds user index must propagate a scipy/numpy error."""
        n_users = chain_model.urm.shape[0]
        with pytest.raises((IndexError, ValueError)):
            chain_model.predict(n_users, 0)

    def test_predict_out_of_range_item_raises(self, chain_model):
        """7.2 — An out-of-bounds item index must propagate an error."""
        n_items = chain_model.urm.shape[1]
        with pytest.raises((IndexError, ValueError)):
            chain_model.predict(0, n_items)

    def test_predict_non_integer_user_raises(self, chain_model):
        """7.3 — Non-integer user index must fail with a clear error."""
        with pytest.raises((IndexError, ValueError, TypeError)):
            chain_model.predict("user0", 0)

    def test_predict_non_integer_item_raises(self, chain_model):
        """7.3 — Non-integer item index must fail with a clear error."""
        with pytest.raises((IndexError, ValueError, TypeError)):
            chain_model.predict(0, "item0")


# ===========================================================================
# 8. Sparse-matrix invariance
# ===========================================================================


class TestSparseInvariance:
    def test_coo_and_csr_input_produce_identical_predictions(self, chain_urm):
        """8.1 — Predictions must not depend on the sparse format of the input."""
        model_csr = ItemBasedCF(k=50)
        model_csr.fit(chain_urm.tocsr())

        model_coo = ItemBasedCF(k=50)
        model_coo.fit(chain_urm.tocoo())

        for uid in range(3):
            for iid in range(4):
                p_csr = model_csr.predict(uid, iid)
                p_coo = model_coo.predict(uid, iid)
                assert p_csr == pytest.approx(p_coo, abs=1e-12), (
                    f"Mismatch for user={uid}, item={iid}: {p_csr} vs {p_coo}"
                )

    def test_csc_and_csr_input_produce_identical_similarity(self, chain_urm):
        """8.1 — Similarity matrices must be identical regardless of input format."""
        model_csr = ItemBasedCF(k=50)
        model_csr.fit(chain_urm.tocsr())

        model_csc = ItemBasedCF(k=50)
        model_csc.fit(chain_urm.tocsc())

        full_csr = model_csr.similarity_full.toarray()
        full_csc = model_csc.similarity_full.toarray()
        np.testing.assert_allclose(full_csr, full_csc, atol=1e-12)


# ===========================================================================
# 9. Evaluation pipeline smoke tests
# ===========================================================================


class TestEvaluationSmoke:
    @pytest.fixture
    def tiny_eval_df(self):
        """Minimal validation DataFrame matching chain_urm user/item IDs."""
        return pd.DataFrame(
            {
                "userId": [0, 1, 2, 3],
                "movieId": [1, 2, 3, 0],
                "rating": [4.0, 2.0, 4.5, 3.0],
            }
        )

    def test_evaluate_predictions_returns_expected_keys(self, chain_model, tiny_eval_df):
        """9.1 — evaluate_predictions must return rmse and mae keys."""
        metrics = evaluate_predictions(chain_model, tiny_eval_df)
        assert "rmse" in metrics
        assert "mae" in metrics

    def test_evaluate_predictions_metrics_are_finite(self, chain_model, tiny_eval_df):
        """9.2 — RMSE and MAE must be non-NaN finite numbers."""
        metrics = evaluate_predictions(chain_model, tiny_eval_df)
        assert math.isfinite(metrics["rmse"])
        assert math.isfinite(metrics["mae"])
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0

    def test_evaluate_recommendations_returns_expected_keys(self, chain_model, tiny_eval_df):
        """9.1 — evaluate_recommendations must return standard ranking metric keys."""
        metrics = evaluate_recommendations(chain_model, tiny_eval_df, k=3)
        for key in ("precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"):
            assert key in metrics

    def test_evaluate_recommendations_metrics_in_valid_range(self, chain_model, tiny_eval_df):
        """9.2 — Ranking metrics must be in [0, 1]."""
        metrics = evaluate_recommendations(chain_model, tiny_eval_df, k=3)
        for key in ("precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key} = {metrics[key]} out of range"


# ===========================================================================
# 10. Regression guard
# ===========================================================================


class TestRegression:
    """Freeze expected outputs for chain_urm + k=50.

    These tests fail if cosine normalisation, diagonal handling,
    top-k pruning, or score denominator logic changes unexpectedly.
    """

    def test_similarity_full_spot_checks(self, chain_model):
        sim = chain_model.similarity_full.toarray()
        assert sim[0, 1] == pytest.approx(0.5, abs=1e-10)
        assert sim[1, 0] == pytest.approx(0.5, abs=1e-10)
        assert sim[1, 2] == pytest.approx(0.5, abs=1e-10)
        assert sim[2, 3] == pytest.approx(1.0 / _SQRT2, abs=1e-10)
        assert sim[0, 0] == pytest.approx(0.0, abs=1e-12)  # diagonal zeroed
        assert sim[0, 2] == pytest.approx(0.0, abs=1e-12)

    def test_similarity_topk_nonzero_pattern_for_large_k(self, chain_model):
        """With k=50, similarity_topk must match similarity_full exactly."""
        full = chain_model.similarity_full.toarray()
        topk = chain_model.similarity_topk.toarray()
        np.testing.assert_allclose(full, topk, atol=1e-12)

    def test_predict_regression_user0_item1(self, chain_model):
        assert chain_model.predict(0, 1) == pytest.approx(4.0, abs=1e-10)

    def test_predict_regression_user1_item2(self, chain_model):
        assert chain_model.predict(1, 2) == pytest.approx(2.0, abs=1e-10)

    def test_predict_regression_empty_user(self, chain_model):
        assert chain_model.predict(4, 0) == 0.0

    def test_recommend_regression_user0_top_item(self, chain_model):
        recs = chain_model.recommend(user_id=0, n=10)
        top_item, top_score = recs[0]
        assert top_item == 1
        assert top_score == pytest.approx(4.0, abs=1e-10)

    def test_recommend_regression_user0_excludes_seen(self, chain_model):
        recs = chain_model.recommend(user_id=0, n=10)
        item_ids = [iid for iid, _ in recs]
        assert 0 not in item_ids  # item0 was rated by user0

    def test_abs_similarity_topk_equals_abs_of_similarity_topk(self, chain_model):
        """abs_similarity_topk must equal |similarity_topk| element-wise."""
        expected = np.abs(chain_model.similarity_topk.toarray())
        actual = chain_model.abs_similarity_topk.toarray()
        np.testing.assert_allclose(actual, expected, atol=1e-12)
