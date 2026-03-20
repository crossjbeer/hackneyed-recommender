from pathlib import Path

import pandas as pd
import scipy.sparse as sp

from hacrec.load import load_ratings_splits, load_user_item_matrix


def test_load_ratings_splits_loads_train_val_test(tmp_path):
    out = tmp_path

    train_df = pd.DataFrame(
        {
            "userId": [0, 0, 1],
            "movieId": [0, 1, 2],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": [1, 2, 3],
        }
    )
    val_df = pd.DataFrame(
        {
            "userId": [0, 1],
            "movieId": [2, 0],
            "rating": [4.0, 2.0],
            "timestamp": [4, 5],
        }
    )
    test_df = pd.DataFrame(
        {
            "userId": [0, 1],
            "movieId": [3, 4],
            "rating": [5.0, 4.0],
            "timestamp": [6, 7],
        }
    )

    train_df.to_csv(out / "train_ratings.csv", index=False)
    val_df.to_csv(out / "val_ratings.csv", index=False)
    test_df.to_csv(out / "test_ratings.csv", index=False)

    loaded_train, loaded_val, loaded_test = load_ratings_splits(out)

    pd.testing.assert_frame_equal(loaded_train, train_df)
    pd.testing.assert_frame_equal(loaded_val, val_df)
    pd.testing.assert_frame_equal(loaded_test, test_df)


def test_load_user_item_matrix_loads_saved_npz(tmp_path):
    out = Path(tmp_path)
    matrix = sp.csr_matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    sp.save_npz(out / "user_item_matrix.npz", matrix)

    loaded = load_user_item_matrix(out)

    diff = (loaded - matrix).nnz
    assert loaded.shape == matrix.shape
    assert diff == 0
