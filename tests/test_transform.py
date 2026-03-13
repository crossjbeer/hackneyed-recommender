"""Tests for hacrec.transform module."""

import textwrap

import pandas as pd
import pytest
import scipy.sparse as sp

from hacrec.transform import (
    build_urm,
    split_ratings,
    transform_movies,
    transform_ratings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ratings_csv(tmp_path):
    """Write a minimal ratings.csv with three users and return (dir, filename).

    User 1: 10 ratings  (at or above a threshold of 10)
    User 2: 10 ratings  (at or above a threshold of 10)
    User 3:  3 ratings  (below a threshold of 10; above a threshold of 3)
    """
    content = textwrap.dedent("""\
        userId,movieId,rating,timestamp
        1,10,4.0,1000000000
        1,20,5.0,1000000100
        1,30,3.0,1000000200
        1,40,4.5,1000000300
        1,50,2.0,1000000400
        1,60,3.5,1000000500
        1,70,4.0,1000000600
        1,80,5.0,1000000700
        1,90,4.0,1000000800
        1,100,3.0,1000000900
        2,10,2.0,1000001000
        2,20,3.0,1000001100
        2,30,4.0,1000001200
        2,40,5.0,1000001300
        2,50,1.0,1000001400
        2,60,4.5,1000001500
        2,70,3.0,1000001600
        2,80,2.5,1000001700
        2,90,4.0,1000001800
        2,100,5.0,1000001900
        3,10,1.0,1000002000
        3,20,2.0,1000002100
        3,30,5.0,1000002200
    """)
    csv_file = tmp_path / "ratings.csv"
    csv_file.write_text(content)
    return tmp_path, "ratings.csv"


@pytest.fixture
def movies_csv(tmp_path):
    """Write a minimal movies.csv and return (dir, filename)."""
    content = textwrap.dedent("""\
        movieId,title,genres
        10,Toy Story (1995),Animation|Children|Comedy
        20,GoldenEye (1995),Action|Adventure|Thriller
        30,Four Rooms (1995),Thriller
    """)
    csv_file = tmp_path / "movies.csv"
    csv_file.write_text(content)
    return tmp_path, "movies.csv"


@pytest.fixture
def simple_ratings_df():
    """A small pre-built ratings DataFrame (already index-mapped).

    User 0: 3 ratings — timestamps 2020-01-01/02/03
    User 1: 4 ratings — timestamps 2020-02-01/02/03/04

    Expected split:
        test  — user0: 2020-01-03, user1: 2020-02-04
        val   — user0: 2020-01-02, user1: 2020-02-03
        train — user0: 2020-01-01, user1: 2020-02-01 + 2020-02-02  (3 rows)
    """
    return pd.DataFrame({
        "userId":    [0,            0,            0,            1,            1,            1,            1           ],
        "movieId":   [0,            1,            2,            0,            1,            2,            3           ],
        "rating":    [5.0,          3.0,          4.0,          2.0,          4.5,          3.5,          5.0         ],
        "timestamp": pd.to_datetime([
            "2020-01-01", "2020-01-02", "2020-01-03",
            "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04",
        ]),
    })


# ---------------------------------------------------------------------------
# transform_ratings
# ---------------------------------------------------------------------------

class TestTransformRatings:
    def test_return_types(self, ratings_csv):
        data_dir, filename = ratings_csv
        df, user_mapping, item_mapping = transform_ratings(data_dir, filename, rating_threshold=10)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(user_mapping, dict)
        assert isinstance(item_mapping, dict)

    def test_required_columns_present(self, ratings_csv):
        data_dir, filename = ratings_csv
        df, _, _ = transform_ratings(data_dir, filename, rating_threshold=10)
        assert {"userId", "movieId", "rating", "timestamp"}.issubset(df.columns)

    def test_timestamp_column_is_datetime(self, ratings_csv):
        data_dir, filename = ratings_csv
        df, _, _ = transform_ratings(data_dir, filename, rating_threshold=10)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_users_below_threshold_are_filtered_out(self, ratings_csv):
        """User 3 has only 3 ratings; threshold=10 should exclude them."""
        data_dir, filename = ratings_csv
        _, user_mapping, _ = transform_ratings(data_dir, filename, rating_threshold=10)
        assert len(user_mapping) == 2  # only users 1 and 2

    def test_users_at_threshold_are_kept(self, ratings_csv):
        """Users 1 and 2 each have exactly 10 ratings; they must be kept."""
        data_dir, filename = ratings_csv
        df, user_mapping, _ = transform_ratings(data_dir, filename, rating_threshold=10)
        assert len(user_mapping) == 2
        assert len(df) == 20  # 10 ratings each

    def test_threshold_lowered_admits_more_users(self, ratings_csv):
        """threshold=3 causes user 3 (with 3 ratings) to be included."""
        data_dir, filename = ratings_csv
        _, user_mapping, _ = transform_ratings(data_dir, filename, rating_threshold=3)
        assert len(user_mapping) == 3

    def test_threshold_one_keeps_all_users(self, ratings_csv):
        data_dir, filename = ratings_csv
        _, user_mapping, _ = transform_ratings(data_dir, filename, rating_threshold=1)
        assert len(user_mapping) == 3

    def test_user_ids_are_zero_based_contiguous(self, ratings_csv):
        data_dir, filename = ratings_csv
        _, user_mapping, _ = transform_ratings(data_dir, filename, rating_threshold=10)
        assert sorted(user_mapping.values()) == list(range(len(user_mapping)))

    def test_item_ids_are_zero_based_contiguous(self, ratings_csv):
        data_dir, filename = ratings_csv
        _, _, item_mapping = transform_ratings(data_dir, filename, rating_threshold=10)
        assert sorted(item_mapping.values()) == list(range(len(item_mapping)))

    def test_dataframe_userid_values_match_mapping(self, ratings_csv):
        """Every userId value in the DataFrame must exist in user_mapping."""
        data_dir, filename = ratings_csv
        df, user_mapping, _ = transform_ratings(data_dir, filename, rating_threshold=10)
        assert set(df["userId"].unique()).issubset(set(user_mapping.values()))

    def test_dataframe_movieid_values_match_mapping(self, ratings_csv):
        data_dir, filename = ratings_csv
        df, _, item_mapping = transform_ratings(data_dir, filename, rating_threshold=10)
        assert set(df["movieId"].unique()).issubset(set(item_mapping.values()))

    def test_all_users_below_threshold_returns_empty(self, ratings_csv):
        """When no user meets the threshold the result should be empty."""
        data_dir, filename = ratings_csv
        df, user_mapping, item_mapping = transform_ratings(data_dir, filename, rating_threshold=100)
        assert df.empty
        assert user_mapping == {}
        assert item_mapping == {}


# ---------------------------------------------------------------------------
# split_ratings
# ---------------------------------------------------------------------------

class TestSplitRatings:
    def test_returns_three_dataframes(self, simple_ratings_df):
        result = split_ratings(simple_ratings_df)
        assert len(result) == 3
        for part in result:
            assert isinstance(part, pd.DataFrame)

    def test_combined_length_equals_input(self, simple_ratings_df):
        """All ratings must appear exactly once across the three splits."""
        train, val, test = split_ratings(simple_ratings_df)
        assert len(train) + len(val) + len(test) == len(simple_ratings_df)

    def test_test_has_one_row_per_user(self, simple_ratings_df):
        _, _, test = split_ratings(simple_ratings_df)
        n_users = simple_ratings_df["userId"].nunique()
        assert len(test) == n_users
        assert test["userId"].nunique() == n_users

    def test_val_has_one_row_per_user(self, simple_ratings_df):
        _, val, _ = split_ratings(simple_ratings_df)
        n_users = simple_ratings_df["userId"].nunique()
        assert len(val) == n_users
        assert val["userId"].nunique() == n_users

    def test_test_contains_most_recent_rating_per_user(self, simple_ratings_df):
        _, _, test = split_ratings(simple_ratings_df)
        for uid in simple_ratings_df["userId"].unique():
            expected_ts = simple_ratings_df.loc[
                simple_ratings_df["userId"] == uid, "timestamp"
            ].max()
            actual_ts = test.loc[test["userId"] == uid, "timestamp"].iloc[0]
            assert actual_ts == expected_ts

    def test_val_contains_second_most_recent_rating_per_user(self, simple_ratings_df):
        _, val, _ = split_ratings(simple_ratings_df)
        for uid in simple_ratings_df["userId"].unique():
            user_ts = (
                simple_ratings_df.loc[simple_ratings_df["userId"] == uid, "timestamp"]
                .sort_values()
            )
            expected_ts = user_ts.iloc[-2]  # second-to-last
            actual_ts = val.loc[val["userId"] == uid, "timestamp"].iloc[0]
            assert actual_ts == expected_ts

    def test_train_does_not_contain_val_or_test_timestamps(self, simple_ratings_df):
        train, val, test = split_ratings(simple_ratings_df)
        held_out_ts = set(val["timestamp"]) | set(test["timestamp"])
        assert not set(train["timestamp"]).intersection(held_out_ts)

    def test_train_row_count(self, simple_ratings_df):
        """Each user contributes max(0, n_ratings - 2) rows to train."""
        train, _, _ = split_ratings(simple_ratings_df)
        expected = sum(
            max(0, count - 2)
            for count in simple_ratings_df.groupby("userId").size()
        )
        assert len(train) == expected

    def test_splits_are_reset_indexed(self, simple_ratings_df):
        for df in split_ratings(simple_ratings_df):
            assert list(df.index) == list(range(len(df)))

    def test_user_with_two_ratings_contributes_nothing_to_train(self):
        """A user with exactly 2 ratings should fill only val and test."""
        df = pd.DataFrame({
            "userId":    [0,                        0           ],
            "movieId":   [0,                        1           ],
            "rating":    [4.0,                      5.0         ],
            "timestamp": pd.to_datetime(["2021-01-01", "2021-06-01"]),
        })
        train, val, test = split_ratings(df)
        assert len(test) == 1
        assert len(val) == 1
        assert len(train) == 0

    def test_temporal_ordering_is_respected(self):
        """Ratings provided out of order must still split on chronological recency."""
        df = pd.DataFrame({
            "userId":    [0,            0,            0           ],
            "movieId":   [2,            0,            1           ],
            "rating":    [3.0,          5.0,          4.0         ],
            # Deliberately out of order — chronological order is movieId 0, 1, 2
            "timestamp": pd.to_datetime(["2021-03-01", "2021-01-01", "2021-02-01"]),
        })
        _, val, test = split_ratings(df)
        assert test.iloc[0]["timestamp"] == pd.Timestamp("2021-03-01")
        assert val.iloc[0]["timestamp"] == pd.Timestamp("2021-02-01")


# ---------------------------------------------------------------------------
# build_urm
# ---------------------------------------------------------------------------

class TestBuildUrm:
    def test_returns_csr_matrix(self, simple_ratings_df):
        shape = (
            int(simple_ratings_df["userId"].max()) + 1,
            int(simple_ratings_df["movieId"].max()) + 1,
        )
        urm = build_urm(simple_ratings_df, shape)
        assert isinstance(urm, sp.csr_matrix)

    def test_shape_matches_argument(self, simple_ratings_df):
        shape = (10, 20)
        urm = build_urm(simple_ratings_df, shape)
        assert urm.shape == shape

    def test_nnz_matches_number_of_ratings(self, simple_ratings_df):
        shape = (
            int(simple_ratings_df["userId"].max()) + 1,
            int(simple_ratings_df["movieId"].max()) + 1,
        )
        urm = build_urm(simple_ratings_df, shape)
        assert urm.nnz == len(simple_ratings_df)

    def test_stored_values_match_ratings(self, simple_ratings_df):
        shape = (
            int(simple_ratings_df["userId"].max()) + 1,
            int(simple_ratings_df["movieId"].max()) + 1,
        )
        urm = build_urm(simple_ratings_df, shape)
        for _, row in simple_ratings_df.iterrows():
            u, i, r = int(row["userId"]), int(row["movieId"]), row["rating"]
            assert urm[u, i] == r

    def test_empty_dataframe_produces_zero_matrix(self):
        empty_df = pd.DataFrame({
            "userId":  pd.Series([], dtype="int64"),
            "movieId": pd.Series([], dtype="int64"),
            "rating":  pd.Series([], dtype="float64"),
        })
        shape = (5, 8)
        urm = build_urm(empty_df, shape)
        assert urm.shape == shape
        assert urm.nnz == 0

    def test_explicit_larger_shape_is_respected(self, simple_ratings_df):
        """The caller can pass a shape larger than the data's index range."""
        shape = (100, 200)
        urm = build_urm(simple_ratings_df, shape)
        assert urm.shape == shape


# ---------------------------------------------------------------------------
# transform_movies
# ---------------------------------------------------------------------------

class TestTransformMovies:
    def test_returns_dict(self, movies_csv):
        data_dir, filename = movies_csv
        result = transform_movies(data_dir, filename)
        assert isinstance(result, dict)

    def test_keys_are_movie_ids(self, movies_csv):
        data_dir, filename = movies_csv
        result = transform_movies(data_dir, filename)
        assert set(result.keys()) == {10, 20, 30}

    def test_values_are_titles(self, movies_csv):
        data_dir, filename = movies_csv
        result = transform_movies(data_dir, filename)
        assert result[10] == "Toy Story (1995)"
        assert result[20] == "GoldenEye (1995)"
        assert result[30] == "Four Rooms (1995)"

    def test_all_movies_present(self, movies_csv):
        data_dir, filename = movies_csv
        result = transform_movies(data_dir, filename)
        assert len(result) == 3

    def test_empty_csv_returns_empty_dict(self, tmp_path):
        csv_file = tmp_path / "movies.csv"
        csv_file.write_text("movieId,title,genres\n")
        result = transform_movies(tmp_path, "movies.csv")
        assert result == {}
