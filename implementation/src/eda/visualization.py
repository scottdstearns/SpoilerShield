"""
Visualization tools for text data analysis.

This module provides the TextVisualizer class for creating various
visualizations of text data, including word clouds, frequency plots,
and class distribution plots.

Here we implement all methods in the TextVisualizer class following the docstrings
and type hints provided. 
"""

from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from .text_analyzer import TextAnalyzer


class TextVisualizer:
    """
    A class for creating visualizations of text data.

    This class provides methods for creating various visualizations
    of text data, including word clouds, frequency plots, and
    class distribution plots.

    Attributes:
        texts (pd.Series): The text data to visualize
        labels (Optional[pd.Series]): Optional labels for the texts
        stop_words (set): Set of stop words to exclude from visualizations
    """   
    def __init__(
        self,
        texts: pd.Series,
        labels: Optional[pd.Series] = None,
        language: str = 'english'
    ) -> None:
        """
        Initialize the TextVisualizer.

        Args:
            texts: A pandas Series containing the text data
            labels: Optional pandas Series containing labels for the texts
            language: Language for stop words (default: 'english')

        Raises:
            ValueError: If texts is not a pandas Series or if labels length
                      doesn't match texts length
        """
        # Using TextAnalyzer for common functionality
        self.analyzer = TextAnalyzer(texts, labels, language)
        
        # Store the texts and labels
        self.texts = texts
        self.labels = labels

    def plot_word_frequencies(
        self,
        n_top: int = 20,
        remove_stopwords: bool = True,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot word frequency distribution.

        Args:
            n_top: Number of top words to plot
            remove_stopwords: Whether to remove stop words
            figsize: Figure size as (width, height)

        Raises:
            ValueError: If labels are not provided
        """
        # Use analyzer's methods to get word frequencies
        word_freq = self.analyzer.analyze_word_frequency(n_top, remove_stopwords)

        # Create the barplot of the top n words
        plt.figure(figsize=figsize)
        sns.barplot(x=word_freq.index, y=word_freq.values, palette='viridis')
        plt.title('Word Frequencies')
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.show()

    def plot_class_distribution(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot the distribution of classes.

        Args:
            figsize: Figure size as (width, height)

        Raises:
            ValueError: If labels are not provided
        """
        # Get the class distribution
        class_counts, class_percentages, imbalance_ratio = self.analyzer.analyze_class_distribution()

        # Create the plot with counts labels
        plt.subplots(figsize=figsize)
        plt.suptitle(f'Class Distribution (Imbalance Ratio: {imbalance_ratio:.2f})', fontsize=14, fontweight='bold',color='navy')
        plt.subplot(1,2,1)
        sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title("Class Counts", fontsize=12, fontweight='bold',color='gray')
        plt.grid(True)

        # Create the plot with percentage labels
        plt.subplot(1,2,2)
        sns.barplot(x=class_percentages.index, y=class_percentages.values, palette='viridis')
        plt.xlabel('Class')
        plt.ylabel('Percentage')
        plt.title("Class Distribution", fontsize=12, fontweight='bold',color='gray')
        plt.grid(True)
        plt.show()

    def plot_wordcloud(
        self,
        class_label: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10),
        **wordcloud_params
    ) -> None:
        """
        Plot a word cloud of the text data.

        Args:
            class_label: Optional class label to filter texts
            figsize: Figure size as (width, height)
            **wordcloud_params: Additional parameters for WordCloud

        Raises:
            ValueError: If class_label is provided but labels are not available
        """
        # Validate inputs
        if class_label is not None and self.labels is None:
            raise ValueError("class_label is provided but labels are not available")
        
        # Filter the texts by class if the class label is provided
        if class_label is not None:
            texts = self.texts[self.labels == class_label]
        else:
            texts = self.texts

        # Generate the word cloud from the combined text of all documents using the analyzer
        wordcloud = self.analyzer.generate_wordcloud(class_label, **wordcloud_params)
       
        # Create the plot
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        if class_label is not None:
            plt.title(f'Word Cloud for {class_label}', fontsize=14, fontweight='bold',color='navy')
        else:
            plt.title('Word Cloud', fontsize=14, fontweight='bold',color='navy')
        plt.show()

        # Note: The word cloud is generated using the analyzer's generate_wordcloud method.
        # To save the image: 
        # image = wordcloud.to_image()
        # image.save('wordcloud.png') 

    def plot_document_length_distribution(
        self,
        bins: int = 50,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot the distribution of document lengths.

        Args:
            bins: Number of bins for the histogram
            figsize: Figure size as (width, height)
        """
        # Get the document lengths using the analyzer's public method
        document_lengths = self.analyzer.get_document_lengths()

        # Create the histogram
        plt.figure(figsize=figsize)
        plt.hist(document_lengths, bins=bins, edgecolor='black')
        plt.axvline(np.mean(document_lengths), color='red', linestyle='--', label='Mean')
        plt.axvline(np.median(document_lengths), color='green', linestyle='--', label='Median')
        plt.title('Document Length Distribution')
        plt.xlabel('Document Length (words)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_class_word_frequencies(
        self,
        n_top: int = 10,
        remove_stopwords: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot word frequencies by class.

        Args:
            n_top: Number of top words to plot per class
            remove_stopwords: Whether to remove stop words
            figsize: Figure size as (width, height)

        Raises:
            ValueError: If labels are not provided
        """
        # Make sure labels are available
        if self.labels is None:
            raise ValueError("labels are not available")
        
        # Get the unique classes
        classes = self.labels.unique()

        # Create a subplot grid (Note this may not look good - revisit sds)
        fig, ax = plt.subplots(1, len(classes), figsize=figsize)

        # For each class:
        for i, class_label in enumerate(classes):
            # Get the texts for this class
            texts = self.texts[self.labels == class_label]
            
            # Calculate word frequencies
            word_freq = self.analyzer.analyze_word_frequency(n_top, remove_stopwords)

            # Create a bar plot for this class
            sns.barplot(x=word_freq.index, y=word_freq.values, ax=ax[i])

            # Title and labels
            ax[i].set_title(f'Top {n_top} Words for {class_label}', fontsize=12, fontweight='bold',color='gray')
            ax[i].set_xlabel('Word', fontsize=10, color='gray')
            ax[i].set_ylabel('Frequency', fontsize=10, color='gray')
            ax[i].grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def save_plot_as(
        self,
        filename: str
    ) -> None:
        """
        Save the current plot as a png file.

        Args:
            filename: Name of the file to save the plot as

        Raises:
            ValueError: If the filename is not a string
        """
        # Check if the filename is a string
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        
        # Save the plot as a png file
        plt.savefig(filename)
        print(f"Plot saved as {filename}")