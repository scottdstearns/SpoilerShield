"""
Data loader for the EDA module. This module provides methods to load the IMDB
movie reviews and movie details datasets.
"""

import pandas as pd

class DataLoader:
    """
    Data loader for the EDA module. This module provides methods to load the IMDB
    movie reviews and movie details datasets.

    Attributes:
        movie_reviews_path: Path to the IMDB movie reviews dataset.
        movie_details_path: Path to the IMDB movie details dataset.
    """
    def __init__(self, movie_reviews_path: str, movie_details_path: str):
        self.movie_reviews_path = movie_reviews_path
        self.movie_details_path = movie_details_path

    def load_imdb_movie_reviews(self) -> pd.DataFrame:
        """
        Load the IMDB movie reviews dataset. This contains the ispoiler
        ground truth labels.
        """
        return pd.read_json(self.movie_reviews_path, lines=True)
    
    def load_imdb_movie_details(self) -> pd.DataFrame:
        """
        Load the IMDB movie details dataset. This contains the movie details
        such as plot summary, plot synopsis, genre, etc.
        """
        return pd.read_json(self.movie_details_path, lines=True)