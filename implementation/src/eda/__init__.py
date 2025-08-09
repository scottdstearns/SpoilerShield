"""
Exploratory Data Analysis (EDA) module for text classification.

This module provides tools for analyzing and visualizing text data,
including text statistics, class distribution analysis, and word frequency analysis.

The module includes:
1. DataLoader: For loading and preprocessing IMDB movie data
2. TextAnalyzer: For analyzing text statistics and patterns
3. TextVisualizer: For creating visualizations of text data
4. TransformerTextProcessor: For preprocessing text data for transformer models
"""

from .text_analyzer import TextAnalyzer
from .visualization import TextVisualizer
from .data_loader import DataLoader
from .transformer_processor import TransformerTextProcessor

__all__ = ['TextAnalyzer', 'TextVisualizer', 'DataLoader', 'TransformerTextProcessor'] 