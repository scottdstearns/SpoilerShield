"""
Script to split IMDB movie reviews data into train/test sets.

This script:
1. Loads the IMDB movie reviews data
2. Splits it into train/test sets while maintaining class distribution
3. Saves the resulting datasets as JSON files
4. Provides options for sampling if needed

Usage:
    python split_data.py [--test_size TEST_SIZE] [--sample_size SAMPLE_SIZE] [--random_state RANDOM_STATE]
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from implementation.src.eda.data_loader import DataLoader

def setup_paths():
    """Set up input and output paths."""
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root_path, 'implementation', 'src', 'data')
    
    # Input paths
    imdb_movie_reviews_path = os.path.join(data_path, 'IMDB_reviews.json')
    
    # Output paths
    # output_dir = os.path.join(data_path, 'split')
    output_dir = data_path
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_reviews.json')
    test_path = os.path.join(output_dir, 'test_reviews.json')
    
    return imdb_movie_reviews_path, train_path, test_path

def load_and_prepare_data(reviews_path, sample_size=None, random_state=42):
    """
    Load and prepare the data for splitting.
    
    Args:
        reviews_path: Path to the reviews JSON file
        sample_size: Optional size to sample from the data
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame containing the reviews data
    """
    print(f"Loading data from {reviews_path}")
    data_loader = DataLoader(reviews_path, None)  # We only need reviews for splitting
    df = data_loader.load_imdb_movie_reviews()
    
    if sample_size is not None:
        print(f"Sampling {sample_size} records while maintaining class distribution")
        df = df.groupby('is_spoiler').apply(
            lambda x: x.sample(
                min(len(x), int(sample_size * len(x) / len(df))),
                random_state=random_state
            )
        ).reset_index(drop=True)
    
    return df

def split_and_save_data(df, train_path, test_path, test_size=0.2, random_state=42):
    """
    Split the data into train/test sets and save them.
    
    Args:
        df: DataFrame containing the reviews data
        train_path: Path to save the training data
        test_path: Path to save the test data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    """
    # Split the data
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['is_spoiler']
    )
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Training set size: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    print("\nClass distribution in training set:")
    print(train_df['is_spoiler'].value_counts(normalize=True))
    print("\nClass distribution in test set:")
    print(test_df['is_spoiler'].value_counts(normalize=True))
    
    # Save the datasets
    print(f"\nSaving training data to {train_path}")
    train_df.to_json(train_path, orient='records', lines=True)
    
    print(f"Saving test data to {test_path}")
    test_df.to_json(test_path, orient='records', lines=True)

def main():
    parser = argparse.ArgumentParser(description='Split IMDB movie reviews data into train/test sets')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--sample_size', type=int, default=None,
                      help='Optional size to sample from the data (default: None)')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Setup paths
    reviews_path, train_path, test_path = setup_paths()
    
    # Load and prepare data
    df = load_and_prepare_data(reviews_path, args.sample_size, args.random_state)
    
    # Split and save data
    split_and_save_data(
        df,
        train_path,
        test_path,
        test_size=args.test_size,
        random_state=args.random_state
    )

if __name__ == '__main__':
    main() 