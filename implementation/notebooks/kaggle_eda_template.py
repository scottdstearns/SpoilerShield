"""
Template for running EDA in Kaggle notebooks.

This template shows how to use the SpoilerShield EDA tools in a Kaggle notebook.
"""

# Install required packages from our Kaggle dataset
# IMPORTANT: We use kaggle_environment_requirements.txt which is specifically
# configured for Kaggle's environment to avoid conflicts with pre-installed packages
#!pip install -q -r /kaggle/input/spoiler-shield-code/kaggle_environment_requirements.txt

# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set project_root for both script and notebook environments
if '__file__' in globals():
    # Script mode
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    # Notebook mode
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# Import our utils
if os.path.exists("/kaggle/input"):  
    sys.path.append('/kaggle/input/spoiler-shield-code/spoiler_shield_code/implementation/src')
    from utils.kaggle_utils import is_kaggle_environment
else:
    from implementation.src.utils.kaggle_utils import is_kaggle_environment
    


# Load the data
if is_kaggle_environment():
    # Set path for notebook environments
    sys.path.append('/kaggle/input/spoiler-shield-code/spoiler_shield_code/implementation/src')

    from eda.text_analyzer import TextAnalyzer
    from eda.visualization import TextVisualizer
    from eda.data_loader import DataLoader

    data_path = "/kaggle/input/spoiler-shield-code/spoiler_shield_code/implementation/src/data/"
    imdb_movie_reviews_path = data_path + 'train_reviews.json'
    # imdb_movie_reviews_path = data_path + 'IMDB_reviews.json'
    imdb_movie_details_path = data_path + 'IMDB_movie_details.json'
else:
    # Local path
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from implementation.src.eda.data_loader import DataLoader
    from implementation.src.eda.text_analyzer import TextAnalyzer
    from implementation.src.eda.visualization import TextVisualizer

    data_path = os.path.join(root_path, 'implementation', 'src', 'data')
    imdb_movie_reviews_path = os.path.join(data_path, 'train_reviews.json')
    # imdb_movie_reviews_path = os.path.join(data_path, 'IMDB_reviews.json')
    imdb_movie_details_path = os.path.join(data_path, 'IMDB_movie_details.json')

# Load the data
print(f"Loading data from {imdb_movie_reviews_path} and {imdb_movie_details_path}")
data_loader = DataLoader(imdb_movie_reviews_path, imdb_movie_details_path)
df_reviews = data_loader.load_imdb_movie_reviews()
df_details = data_loader.load_imdb_movie_details()

# Initialize the analyzer and visualizer with the data
analyzer = TextAnalyzer(
    texts=df_reviews['review_text'],
    labels=df_reviews['is_spoiler']
)
visualizer = TextVisualizer(
    texts=df_reviews['review_text'],  
    labels=df_reviews['is_spoiler']
)

# Example EDA tasks
# 1. Analyze word frequencies
word_freq = analyzer.analyze_word_frequency(n_top=20, remove_stopwords=True)
print("Top 20 most frequent words:")
print(word_freq)

# 2. Plot word frequencies
visualizer.plot_word_frequencies(n_top=20, remove_stopwords=True)

# 3. Analyze class distribution
class_counts, class_percentages, imbalance_ratio = analyzer.analyze_class_distribution()
print("\nClass distribution:")
print(f"Imbalance ratio: {imbalance_ratio:.2f}")
print("\nClass counts:")
print(class_counts)
print("\nClass percentages:")
print(class_percentages)

# 4. Plot class distribution
visualizer.plot_class_distribution()

# 5. Generate word clouds
# For all data
visualizer.plot_wordcloud()

# For each class (if you have labels)
if df_reviews['is_spoiler'].nunique() > 1:  
    for label in df_reviews['is_spoiler'].unique():
        visualizer.plot_wordcloud(class_label=label)

# 6. Analyze document lengths
document_lengths = analyzer.get_document_lengths()
print("\nDocument length statistics:")
print(f"Mean length: {np.mean(document_lengths):.2f} words")
print(f"Median length: {np.median(document_lengths):.2f} words")
print(f"Min length: {np.min(document_lengths)} words")
print(f"Max length: {np.max(document_lengths)} words")

# 7. Plot document length distribution
visualizer.plot_document_length_distribution()

# 8. Plot word frequencies by class
if df_reviews['is_spoiler'].nunique() > 1:  
    visualizer.plot_class_word_frequencies(n_top=10, remove_stopwords=True) 
    