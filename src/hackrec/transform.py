"""Transforming MovieLens dataset for recommendation"""

DATA_DIR = "data/ml-latest-small"

import pandas as pd 
import pathlib as path 

def save_mapping(mapping: dict, filename: str) -> None:
    """Save a mapping dictionary to a CSV file."""
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['original_id', 'mapped_id'])
    mapping_df.to_csv(filename, index=False)

###
# Ratings
###
def transform_ratings(data_dir: str, filename: str, rating_threshold: int=50) -> pd.DataFrame:
    """Transform the ratings data from the MovieLens dataset."""
    ratings_path = path.Path(data_dir) / filename
    ratings_df = pd.read_csv(ratings_path, header=0)
    # Convert timestamp to datetime
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    
    print(f"Original ratings count: {len(ratings_df)}")

    # Filter out users with fewer than rating_threshold ratings
    user_counts = ratings_df['userId'].value_counts()

    valid_users = user_counts[user_counts >= rating_threshold].index
    ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
    print(f"Filtered ratings count: {len(ratings_df)}")

    # Build user and item mappings 
    user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
    item_mapping = {item_id: idx for idx, item_id in enumerate(ratings_df['movieId'].unique())}
    
    # Map userId and movieId to indices 
    ratings_df['userId'] = ratings_df['userId'].map(user_mapping)
    ratings_df['movieId'] = ratings_df['movieId'].map(item_mapping)

    print(len(user_mapping), "unique users")
    print(len(item_mapping), "unique items")
    print("Sample transformed ratings:")
    print(ratings_df.head())

    return ratings_df, user_mapping, item_mapping 

def transform(): 
    ratings = transform_ratings(DATA_DIR, 'ratings.csv')

    return {"ratings": ratings}

def main(): 
    transformed_data = transform()
    ratings_df, user_mapping, item_mapping = transformed_data["ratings"]
    ratings_df.to_csv("transformed_ratings.csv", index=False)
    save_mapping(user_mapping, "user_mapping.csv")
    save_mapping(item_mapping, "item_mapping.csv")


if __name__ == "__main__":
    main() 