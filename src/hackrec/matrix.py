"""Build the Sparse Interaction Matrix (URM) from the transformed ratings data."""

import pandas as pd
import scipy.sparse as sp
import pathlib as path 

from transform import transform_ratings, DATA_DIR, OUT_DIR, ensure_dir

def build_urm(ratings_df: pd.DataFrame) -> sp.csr_matrix:
    """Build a sparse user-item interaction matrix (URM) from the ratings DataFrame."""
    num_users = ratings_df['userId'].nunique()
    num_items = ratings_df['movieId'].nunique()
    
    print(f"Building URM with {num_users} users and {num_items} items...")
    
    # Create a sparse matrix with shape (num_users, num_items)
    urm = sp.csr_matrix((ratings_df['rating'], (ratings_df['userId'], ratings_df['movieId'])), 
                        shape=(num_users, num_items))
    
    print("URM construction complete.")
    return urm

def main():
    ratings_df, _, _ = transform_ratings(DATA_DIR, 'ratings.csv')
    urm = build_urm(ratings_df)
    ensure_dir(path.Path("./"), OUT_DIR)
    sp.save_npz(path.Path(OUT_DIR)/"user_item_matrix.npz", urm)
    print("URM saved to user_item_matrix.npz")

if __name__ == "__main__":
    main()
