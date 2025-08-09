"""
Example usage of the enhanced TransformerTextProcessor.

This script demonstrates how to use the TransformerTextProcessor
with the SpoilerShield project data for spoiler detection.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the src directory to the path for imports
# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
# Go up to the src directory (this script is in src/eda/)
src_dir = script_dir.parent
# Add src directory to path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from eda.transformer_processor import TransformerTextProcessor
from eda.data_loader import DataLoader
from utils.env_config import EnvConfig


def main():
    """Demonstrate the TransformerTextProcessor functionality."""
    print("=" * 60)
    print("TRANSFORMER TEXT PROCESSOR DEMO")
    print("=" * 60)
    
    # Initialize environment configuration
    config = EnvConfig()
    print(f"Environment: {config.env}")
    
    # Initialize the processor
    print("\n1. Initializing TransformerTextProcessor...")
    # processor = TransformerTextProcessor(
    #     model_name='bert-base-uncased',
    #     max_length=256
    # )
    processor = TransformerTextProcessor(
        model_name='allenai/longformer-base-4096',
        max_length=1024  # or higher as needed
    )
    print(f"   {processor}")
    print(f"   Vocabulary size: {processor.get_vocab_size()}")
    
    # Show special tokens
    print("\n2. Special tokens:")
    special_tokens = processor.get_special_tokens()
    for token_name, token_value in special_tokens.items():
        print(f"   {token_name}: {token_value}")
    
    # Load sample data
    print("\n3. Loading sample data...")
    try:
        data_loader = DataLoader(
            config.get_data_path('train_reviews.json'),
            config.get_data_path('IMDB_movie_details.json')
        )
        df_reviews = data_loader.load_imdb_movie_reviews()
        
        # Use first 10 reviews as sample
        sample_texts = df_reviews['review_text'].head(10)
        sample_labels = df_reviews['is_spoiler'].head(10)
        
        print(f"   Loaded {len(sample_texts)} sample reviews")
        print(f"   Sample review length: {len(sample_texts.iloc[0])} characters")
        
    except Exception as e:
        print(f"   Warning: Could not load data ({str(e)})")
        print("   Using dummy data instead...")
        
        # Create dummy data for demonstration
        sample_texts = pd.Series([
            "This movie was amazing! The plot twist at the end was incredible.",
            "I loved the special effects, but the story was predictable.",
            "The main character dies in the final scene - what a shocking ending!",
            "Great acting and cinematography throughout the film.",
            "The villain turns out to be the hero's brother all along."
        ])
        sample_labels = pd.Series([0, 0, 1, 0, 1])  # 0=not spoiler, 1=spoiler
    
    # Get text statistics
    print("\n4. Text statistics:")
    stats = processor.get_text_statistics(sample_texts)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Encode texts
    print("\n5. Encoding texts...")
    encoded = processor.encode_texts(sample_texts)
    print(f"   Encoded shape: {encoded['input_ids'].shape}")
    print(f"   Keys in encoded output: {list(encoded.keys())}")
    
    # Show first encoded example
    print(f"\n6. First encoded example:")
    print(f"   Original text: {sample_texts.iloc[0][:100]}...")
    print(f"   Token IDs: {encoded['input_ids'][0][:20]}...")
    print(f"   Attention mask: {encoded['attention_mask'][0][:20]}...")
    
    # Decode back to text
    print("\n7. Decoding example:")
    decoded = processor.decode_texts(encoded['input_ids'][0])
    print(f"   Decoded text: {decoded[:100]}...")
    
    # Batch processing example
    print("\n8. Batch processing:")
    batches = processor.batch_encode_texts(
        sample_texts, 
        batch_size=3,
        return_tensors='pt'
    )
    print(f"   Number of batches: {len(batches)}")
    for i, batch in enumerate(batches):
        print(f"   Batch {i+1} shape: {batch['input_ids'].shape}")
    
    # Different output formats
    print("\n9. Different output formats:")
    
    # NumPy arrays
    encoded_np = processor.encode_texts(sample_texts[:2], return_tensors='np')
    print(f"   NumPy format: {type(encoded_np['input_ids'])} {encoded_np['input_ids'].shape}")
    
    # Python lists
    encoded_list = processor.encode_texts(sample_texts[:2], return_tensors=None)
    print(f"   List format: {type(encoded_list['input_ids'])} length {len(encoded_list['input_ids'])}")
    
    # Flexible input handling
    print("\n10. Flexible input handling:")
    
    # Single string
    single_encoded = processor.encode_texts("This is a single review.")
    print(f"   Single string shape: {single_encoded['input_ids'].shape}")
    
    # List of strings
    list_encoded = processor.encode_texts(["Review 1", "Review 2"])
    print(f"   List of strings shape: {list_encoded['input_ids'].shape}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Use the processor to encode your text data")
    print("2. Train a transformer model on the encoded data")
    print("3. Compare performance with the TF-IDF baseline")
    print("4. Experiment with different transformer models")


if __name__ == "__main__":
    main() 