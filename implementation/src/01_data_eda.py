"""
01_data_eda.py - Data Loading and Exploratory Data Analysis

This script implements the first two milestones of the SpoilerShield project:
1. Details of the Dataset
2. Exploratory Data Analysis (EDA)

It also prepares and saves the preprocessed data for model training.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, Any, Tuple

# Add the src directory to the path for imports
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from eda.data_loader import DataLoader
from eda.text_analyzer import TextAnalyzer
from eda.visualization import TextVisualizer
from eda.transformer_processor import TransformerTextProcessor
from utils.env_config import EnvConfig


def load_data(config: EnvConfig) -> pd.DataFrame:
    """
    Load the SpoilerShield dataset.
    
    Args:
        config: Environment configuration
        
    Returns:
        DataFrame containing the reviews data
    """
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)
    
    # Load data using DataLoader
    data_loader = DataLoader(
        config.get_data_path('train_reviews.json'),
        config.get_data_path('IMDB_movie_details.json')
    )
    
    df_reviews = data_loader.load_imdb_movie_reviews()
    df_details = data_loader.load_imdb_movie_details()
    
    print(f"✅ Loaded {len(df_reviews)} reviews")
    print(f"✅ Loaded {len(df_details)} movie details")
    
    # Basic data info
    print(f"\nReview columns: {list(df_reviews.columns)}")
    print(f"Details columns: {list(df_details.columns)}")
    
    # Check for missing values
    print(f"\nMissing values in reviews:")
    print(df_reviews.isnull().sum())
    
    # Clean data
    initial_count = len(df_reviews)
    df_reviews = df_reviews.dropna(subset=['review_text', 'is_spoiler'])
    final_count = len(df_reviews)
    
    if initial_count != final_count:
        print(f"⚠️  Removed {initial_count - final_count} rows with missing values")
    
    print(f"✅ Final dataset size: {final_count} reviews")
    
    return df_reviews, df_details


def analyze_dataset_details(df_reviews: pd.DataFrame, df_details: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze dataset specifications and structure.
    
    Args:
        df_reviews: Reviews DataFrame
        df_details: Movie details DataFrame
        
    Returns:
        Dictionary containing dataset analysis
    """
    print("\n" + "=" * 60)
    print("2. DATASET DETAILS & SPECIFICATIONS")
    print("=" * 60)
    
    # Basic statistics
    total_reviews = len(df_reviews)
    total_movies = len(df_details)
    
    print(f"📊 Total Reviews: {total_reviews:,}")
    print(f"📊 Total Movies: {total_movies:,}")
    
    # Class distribution
    class_distribution = df_reviews['is_spoiler'].value_counts()
    class_percentages = df_reviews['is_spoiler'].value_counts(normalize=True) * 100
    
    print(f"\n📈 Class Distribution:")
    print(f"  Non-spoiler (0): {class_distribution[0]:,} ({class_percentages[0]:.1f}%)")
    print(f"  Spoiler (1): {class_distribution[1]:,} ({class_percentages[1]:.1f}%)")
    
    # Imbalance ratio
    imbalance_ratio = class_distribution.max() / class_distribution.min()
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # Text length statistics
    text_lengths = df_reviews['review_text'].str.len()
    
    print(f"\n📝 Text Statistics (characters):")
    print(f"  Average length: {text_lengths.mean():.0f}")
    print(f"  Median length: {text_lengths.median():.0f}")
    print(f"  Min length: {text_lengths.min()}")
    print(f"  Max length: {text_lengths.max()}")
    print(f"  Std deviation: {text_lengths.std():.0f}")
    
    # Data types
    print(f"\n🔍 Data Types:")
    print(df_reviews.dtypes)
    
    # Sample data
    print(f"\n📄 Sample Reviews:")
    for i in range(min(3, len(df_reviews))):
        review = df_reviews.iloc[i]
        print(f"  Example {i+1} (Spoiler: {review['is_spoiler']}):")
        print(f"    {review['review_text'][:200]}...")
        print()
    
    return {
        'total_reviews': total_reviews,
        'total_movies': total_movies,
        'class_distribution': class_distribution.to_dict(),
        'class_percentages': class_percentages.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'text_length_stats': {
            'mean': text_lengths.mean(),
            'median': text_lengths.median(),
            'min': text_lengths.min(),
            'max': text_lengths.max(),
            'std': text_lengths.std()
        }
    }


def perform_eda(df_reviews: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        df_reviews: Reviews DataFrame
        
    Returns:
        Dictionary containing EDA results
    """
    print("\n" + "=" * 60)
    print("3. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer and visualizer
    analyzer = TextAnalyzer(
        texts=df_reviews['review_text'],
        labels=df_reviews['is_spoiler']
    )
    
    visualizer = TextVisualizer(
        texts=df_reviews['review_text'],
        labels=df_reviews['is_spoiler']
    )
    
    # Get text statistics
    print("\n📊 Text Statistics:")
    text_stats = analyzer.get_text_statistics()
    for key, value in text_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Class distribution analysis
    print("\n🎯 Class Distribution Analysis:")
    class_counts, class_percentages, imbalance_ratio = analyzer.analyze_class_distribution()
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    print("  Class counts:")
    for class_val, count in class_counts.items():
        print(f"    {class_val}: {count}")
    
    # Word frequency analysis
    print("\n📝 Word Frequency Analysis:")
    word_freq = analyzer.analyze_word_frequency(n_top=20, remove_stopwords=True)
    print("  Top 20 most frequent words:")
    for word, freq in word_freq.head(10).items():
        print(f"    {word}: {freq}")
    
    # Document length analysis
    print("\n📏 Document Length Analysis:")
    doc_lengths = analyzer.get_document_lengths()
    print(f"  Average document length: {np.mean(doc_lengths):.1f} words")
    print(f"  Median document length: {np.median(doc_lengths):.1f} words")
    print(f"  Min document length: {np.min(doc_lengths)} words")
    print(f"  Max document length: {np.max(doc_lengths)} words")
    
    # Generate visualizations
    print("\n🎨 Generating Visualizations...")
    
    # Word frequency plot
    visualizer.plot_word_frequencies(n_top=20, remove_stopwords=True)
    
    # Class distribution plot
    visualizer.plot_class_distribution()
    
    # Document length distribution
    visualizer.plot_document_length_distribution()
    
    # Word clouds
    visualizer.plot_wordcloud()
    
    # Word clouds by class
    for label in df_reviews['is_spoiler'].unique():
        visualizer.plot_wordcloud(class_label=label)
    
    # Class word frequencies
    visualizer.plot_class_word_frequencies(n_top=15, remove_stopwords=True)
    
    print("✅ EDA Complete!")
    
    return {
        'text_statistics': text_stats,
        'class_analysis': {
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_percentages.to_dict(),
            'imbalance_ratio': imbalance_ratio
        },
        'word_frequency': word_freq.to_dict(),
        'document_lengths': {
            'mean': np.mean(doc_lengths),
            'median': np.median(doc_lengths),
            'min': np.min(doc_lengths),
            'max': np.max(doc_lengths),
            'std': np.std(doc_lengths)
        }
    }


def prepare_transformer_data(df_reviews: pd.DataFrame, 
                           model_name: str = 'bert-base-uncased',
                           max_length: int = 512) -> Dict[str, Any]:
    """
    Prepare data for transformer model training.
    
    Args:
        df_reviews: Reviews DataFrame
        model_name: Name of the transformer model
        max_length: Maximum sequence length
        
    Returns:
        Dictionary containing preparation results
    """
    print("\n" + "=" * 60)
    print("4. TRANSFORMER DATA PREPARATION")
    print("=" * 60)
    
    # Initialize processor
    processor = TransformerTextProcessor(
        model_name=model_name,
        max_length=max_length
    )
    
    print(f"🤖 Using model: {model_name}")
    print(f"📏 Max sequence length: {max_length}")
    
    # Analyze sequence lengths
    print("\n📊 Analyzing sequence lengths...")
    processor.print_sequence_analysis(df_reviews['review_text'])
    
    # Get texts and labels
    texts = df_reviews['review_text'].tolist()
    labels = df_reviews['is_spoiler'].tolist()
    
    print(f"\n⚙️ Encoding {len(texts)} texts...")
    
    # Encode texts
    encoded = processor.encode_texts(
        texts=texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"✅ Encoded shape: {encoded['input_ids'].shape}")
    print(f"✅ Keys in encoded output: {list(encoded.keys())}")
    
    # Create labels tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Prepare data dictionary
    processed_data = {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': labels_tensor,
        'model_name': model_name,
        'max_length': max_length,
        'num_samples': len(texts)
    }
    
    print(f"✅ Data prepared for model training")
    print(f"   - Input IDs shape: {processed_data['input_ids'].shape}")
    print(f"   - Attention mask shape: {processed_data['attention_mask'].shape}")
    print(f"   - Labels shape: {processed_data['labels'].shape}")
    
    return processed_data


def save_processed_data(processed_data: Dict[str, Any], 
                       metadata: Dict[str, Any],
                       config: EnvConfig) -> None:
    """
    Save processed data for model training.
    
    Args:
        processed_data: Processed transformer data
        metadata: Analysis metadata
        config: Environment configuration
    """
    print("\n" + "=" * 60)
    print("5. SAVING PROCESSED DATA")
    print("=" * 60)
    
    # Create output directory
    output_dir = config.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Save processed data
    data_path = output_dir / 'processed_data.pt'
    torch.save(processed_data, data_path)
    print(f"✅ Saved processed data to: {data_path}")
    
    # Save metadata
    metadata_path = output_dir / 'data_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✅ Saved metadata to: {metadata_path}")
    
    # Save summary
    summary = {
        'data_shape': {
            'input_ids': list(processed_data['input_ids'].shape),
            'attention_mask': list(processed_data['attention_mask'].shape),
            'labels': list(processed_data['labels'].shape)
        },
        'model_info': {
            'model_name': processed_data['model_name'],
            'max_length': processed_data['max_length'],
            'num_samples': processed_data['num_samples']
        },
        'class_distribution': metadata['dataset_details']['class_distribution'],
        'files_created': [
            str(data_path),
            str(metadata_path)
        ]
    }
    
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to: {summary_path}")


def main():
    """Main execution function."""
    print("🎬 SPOILERSHIELD - DATA LOADING & EDA")
    print("=" * 60)
    
    # Initialize configuration
    config = EnvConfig()
    print(f"Environment: {config.env}")
    print(f"Output directory: {config.output_dir}")
    
    # Step 1: Load data
    df_reviews, df_details = load_data(config)
    
    # Step 2: Analyze dataset details
    dataset_details = analyze_dataset_details(df_reviews, df_details)
    
    # Step 3: Perform EDA
    eda_results = perform_eda(df_reviews)
    
    # Step 4: Prepare transformer data
    processed_data = prepare_transformer_data(
        df_reviews,
        model_name='bert-base-uncased',  # Change to 'allenai/longformer-base-4096' for longer sequences
        max_length=512
    )
    
    # Step 5: Save processed data
    metadata = {
        'dataset_details': dataset_details,
        'eda_results': eda_results,
        'processing_timestamp': pd.Timestamp.now().isoformat()
    }
    
    save_processed_data(processed_data, metadata, config)
    
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated visualizations")
    print("2. Check the processed data files")
    print("3. Run 02_model_training.py to train models")
    print("4. Copy working code to notebooks when ready")


if __name__ == "__main__":
    main() 