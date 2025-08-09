"""
Text analysis tools for exploratory data analysis.

This module provides the TextAnalyzer class for analyzing text data,
including basic statistics, class distribution, and word frequency analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


class TextAnalyzer:
    """
    A class for analyzing text data in a classification context.

    This class provides methods for analyzing text statistics, class distribution,
    word frequencies, and generating word clouds. It's designed to work with
    pandas Series of text data and optional labels.

    Attributes:
        texts (pd.Series): The text data to analyze
        labels (Optional[pd.Series]): Optional labels for the texts
        stop_words (set): Set of stop words to exclude from analysis
    """

    def __init__(
        self,
        texts: pd.Series,
        labels: Optional[pd.Series] = None,
        language: str = 'english'
    ) -> None:
        """
        Initialize the TextAnalyzer.

        Args:
            texts: A pandas Series containing the text data
            labels: Optional pandas Series containing labels for the texts
            language: Language for stop words (default: 'english')

        Raises:
            ValueError: If texts is not a pandas Series or if labels length
                      doesn't match texts length
        """
        self._validate_input(texts, labels)

        # Download required NLTK data
        self._download_nltk_data()

        self.texts = texts
        self.labels = labels
        self._tokenized_texts = self._tokenized_texts(texts)
        self.stop_words = self._extended_stop_words(set(stopwords.words(language)))
    
    def _download_nltk_data(self) -> None:
        """
        Download required NLTK data.
        """
        nltk.download('punkt')
        nltk.download('stopwords')

    def _validate_input(
        self,
        texts: pd.Series,
        labels: Optional[pd.Series]
    ) -> None:
        """
        Validate input data.

        Args:
            texts: Text data to validate
            labels: Optional labels to validate

        Raises:
            ValueError: If validation fails

        1. Check if texts is a pandas Series
        2. If labels are provided:
           - Check if labels is a pandas Series
           - Check if lengths match
        """      
        # Check if texts is a pandas Series
        if not isinstance(texts, pd.Series):
            raise ValueError("texts must be a pandas Series")
        
        # Check if labels is a pandas Series
        if labels is not None:
            if not isinstance(labels, pd.Series):
                raise ValueError("labels must be a pandas Series")
            if len(texts) != len(labels):
                raise ValueError("texts and labels must have the same length")

    def _tokenized_texts(self, texts:pd.Series) -> List[List[str]]:
        """
        Tokenize the text data.
        """   
        return [ word_tokenize(text) for text in texts ]
    
    def _extended_stop_words(self, stop_words:set) -> set:
        """
        Extend the stop words with additional words. Avoid stopping contractions
        that could change the meaning of the text (eg "'ve", "n't")
        """
        additional_stop_words = ['@',"'",'.','"','/','!',',',"...",'$']
        # based on running the EDA notebook, we should extend the stop words with the following:
        additional_stop_words.extend(["'s","'n't","(",")","``","''","?",":	",":","xx"])
        return stop_words.union(additional_stop_words)  
                                 

    def get_text_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Calculate basic statistics about the text data.

        Returns:
            Dictionary (keys are strings, values are ints or floats) containing:
            - total_documents: Total number of documents
            - avg_length: Average document length in words
            - min_length: Minimum document length
            - max_length: Maximum document length
            - vocabulary_size: Number of unique words
        """

        # Calculate document lengths
        lengths = [ len(doc) for doc in self._tokenized_texts ]
        
        # Calculate the vocabulary size
        vocabulary = set(word for doc in self._tokenized_texts for word in doc)

        # Initialize the dictionary to store the statistics
        statistics = {
            'total_documents': len(self.texts),
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths), 
            'vocabulary_size': len(vocabulary)
        }
        
        return statistics

    def analyze_class_distribution(self) -> Tuple[pd.Series, pd.Series, float]:
        """
        Analyze the distribution of classes in the dataset.

        Returns:
            Tuple containing:
            - class_counts: Series with count of each class
            - class_percentages: Series with percentage of each class
            - imbalance_ratio: Ratio of most common to least common class

        Raises:
            ValueError: If labels are not provided
        """        
        # Check if labels are available
        if self.labels is None:
            raise ValueError("labels are not available")
        
        # Calculate class counts
        class_counts = self.labels.value_counts()
        
        # Calculate class percentages
        class_percentages = class_counts / class_counts.sum()

        # Calculate the imbalance ratio
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        return class_counts, class_percentages, imbalance_ratio

    def analyze_word_frequency(
        self,
        n_top: int = 20,
        remove_stopwords: bool = True
    ) -> pd.Series:
        """
        Analyze word frequencies in the text data.

        Args:
            n_top: Number of top words to return
            remove_stopwords: Whether to remove stop words

        Returns:
            Series containing word frequencies
        """
        # Validate input
        if not n_top > 0:
            raise ValueError("n_top must be positive.")
        
        # Combine all texts
        all_texts = ' '.join(self.texts)
        
        # Tokenize the combined text
        tokenized_text = word_tokenize(all_texts)

        # Remove stop words if requested
        if remove_stopwords:
            tokenized_text = [ word for word in tokenized_text if word not in self.stop_words ]
        
        # Count the frequencies of the words
        word_freq = Counter(tokenized_text)
        
        # Return the top n words
        return pd.Series(word_freq).sort_values(ascending=False).head(n_top)
    
    def generate_wordcloud(
        self,
        class_label: Optional[str] = None,
        **wordcloud_params
    ) -> WordCloud:
        """
        Generate a word cloud from the text data. The word cloud is generated from
        the combined text of all documents. If a class label is provided, the word cloud
        is generated from the combined text of all documents in the class.

        Args:
            class_label: Optional class label to filter texts
            **wordcloud_params: Additional parameters for WordCloud

        Returns:
            WordCloud object

        Raises:
            ValueError: If class_label is provided but labels are not available
        """
        # Validate inputs 
        if class_label is not None and self.labels is None: 
            raise ValueError("class_label is provided but labels are not available")
        
        # Filter texts by class if specified
        if class_label is not None:
            texts = self.texts[self.labels == class_label]
        else:
            texts = self.texts

        # Combine texts
        all_texts = ' '.join(texts)

        # Generate the word cloud
        wordcloud = WordCloud(**wordcloud_params).generate(all_texts)
        
        return wordcloud

    def get_document_lengths(self) -> List[int]:
        """
        Get the length of each document in words.

        Returns:
            List[int]: List of document lengths
        """
        return [len(doc) for doc in self._tokenized_texts]
