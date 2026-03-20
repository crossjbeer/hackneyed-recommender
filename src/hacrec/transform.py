"""Transforming MovieLens dataset for recommendation experimentation."""

DATA_DIR = "data/"
MOVIELENS_DIR = "ml-latest-small/"
OUT_DIR = "out/"

import pandas as pd
import scipy.sparse as sp
import pathlib as path
from .load import save_mapping
from .util import ensure_dir

###
# Ratings
###
def transform_ratings(data_dir: str, filename: str, rating_threshold: int=10) -> pd.DataFrame:
    """Transform the ratings data from the MovieLens dataset."""
    ratings_path = path.Path(data_dir) / filename
    ratings_df = pd.read_csv(ratings_path, header=0)
    # Convert timestamp to datetime
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    
    # Filter out users with fewer than rating_threshold ratings
    user_counts = ratings_df['userId'].value_counts()
    valid_users = user_counts[user_counts >= rating_threshold].index
    ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]

    # Build user and item mappings 
    user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
    item_mapping = {item_id: idx for idx, item_id in enumerate(ratings_df['movieId'].unique())}
    
    # Map userId and movieId to indices 
    ratings_df['userId'] = ratings_df['userId'].map(user_mapping)
    ratings_df['movieId'] = ratings_df['movieId'].map(item_mapping)

    return ratings_df, user_mapping, item_mapping 

def split_ratings(ratings_df: pd.DataFrame) -> tuple:
    """Split ratings into train, validation, and test sets using per-user temporal order.

    For each user:
      - validation: their second-most recent rating
      - test:       their most recent rating
      - train:      all remaining ratings

    Returns (train_df, val_df, test_df).
    """
    ratings_df = ratings_df.sort_values(['userId', 'timestamp'])

    # Last rating per user → test
    test_idx = ratings_df.groupby('userId').tail(1).index
    remaining_df = ratings_df.drop(index=test_idx)

    # Second-to-last rating per user → validation
    val_idx = remaining_df.groupby('userId').tail(1).index
    train_df = remaining_df.drop(index=val_idx)

    test_df = ratings_df.loc[test_idx]
    val_df = ratings_df.loc[val_idx]

    print(f"Split sizes — train: {len(train_df)}, test: {len(test_df)}, val: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def build_urm(ratings_df: pd.DataFrame, shape: tuple) -> sp.csr_matrix:
    """Build a sparse user-item interaction matrix (URM) from a ratings DataFrame.

    Args:
        ratings_df: DataFrame with mapped userId, movieId, and rating columns.
        shape: (num_users, num_items) — must be derived from the full dataset so
               indices in test/val rows remain valid.
    """
    num_users, num_items = shape
    print(f"Building URM with shape ({num_users} users x {num_items} items) "
          f"from {len(ratings_df)} interactions...")

    urm = sp.csr_matrix(
        (ratings_df['rating'], (ratings_df['userId'], ratings_df['movieId'])),
        shape=(num_users, num_items),
    )

    print("URM construction complete.")
    return urm

### 
# Movies
###
def transform_movies(data_dir: str, filename: str) -> pd.DataFrame: 
    """Builds a simple movie id -> title mapping from the movies.csv file."""
    movies_path = path.Path(data_dir) / filename
    movies_df = pd.read_csv(movies_path, header=0)
    # Map the movieId to the title: 
    movie_mapping = dict(zip(movies_df['movieId'], movies_df['title']))
    return movie_mapping

def main():
    ratings_df, user_mapping, item_mapping = transform_ratings(path.Path(DATA_DIR) / MOVIELENS_DIR, 'ratings.csv')
    
    print(len(user_mapping), "unique users")
    print(len(item_mapping), "unique items")
    print("Sample transformed ratings:")
    print(ratings_df.head())

    # Derive full-dataset shape before splitting so all mapped indices stay valid.
    num_users = int(ratings_df['userId'].max()) + 1
    num_items = int(ratings_df['movieId'].max()) + 1

    train_df, val_df, test_df = split_ratings(ratings_df)

    out = ensure_dir(path.Path("./"), OUT_DIR)
    urm = build_urm(train_df, shape=(num_users, num_items))
    
    sp.save_npz(out / "user_item_matrix.npz", urm)
    train_df.to_csv(out / "train_ratings.csv", index=False)
    test_df.to_csv(out / "test_ratings.csv", index=False)
    val_df.to_csv(out / "val_ratings.csv", index=False)
    save_mapping(user_mapping, out / "user_mapping.csv")
    save_mapping(item_mapping, out / "item_mapping.csv")
    print("Saved: user_item_matrix.npz, train_ratings.csv, test_ratings.csv, val_ratings.csv, user_mapping.csv, item_mapping.csv")

    movie_mapping = transform_movies(path.Path(DATA_DIR) / MOVIELENS_DIR, 'movies.csv')
    save_mapping(movie_mapping, out / "movie_mapping.csv")
    print("Saved: movie_mapping.csv")



if __name__ == "__main__":
    main()