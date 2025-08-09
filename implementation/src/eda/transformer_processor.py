"""
Transformer text processor for BERT-based text classification.

This module provides the TransformerTextProcessor class for preprocessing
text data for transformer models, specifically designed for the SpoilerShield
project's spoiler detection task.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch
import logging

logger = logging.getLogger(__name__)


class TransformerTextProcessor:
    """
    A class for processing text data for transformer models.
    
    This class handles tokenization, encoding, and preprocessing of text data
    for BERT-based models. It supports various output formats and includes
    validation and error handling.
    
    Attributes:
        tokenizer: The transformer tokenizer
        max_length: Maximum sequence length for tokenization
        model_name: Name of the transformer model
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased', 
                 max_length: int = 256,
                 cache_dir: Optional[str] = None) -> None:
        """
        Initialize the TransformerTextProcessor.
        
        Args:
            model_name: Name of the transformer model (default: 'bert-base-uncased')
            max_length: Maximum sequence length for tokenization (default: 256)
            cache_dir: Directory to cache the tokenizer (optional)
            
        Raises:
            ValueError: If model_name is invalid or max_length is not positive
            RuntimeError: If tokenizer fails to load
        """
        self._validate_init_params(model_name, max_length)
        
        self.model_name = model_name
        self.max_length = max_length
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                use_fast=True  # Use fast tokenizer when available
            )
            logger.info(f"Loaded tokenizer for {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for {model_name}: {str(e)}")
    
    def _validate_init_params(self, model_name: str, max_length: int) -> None:
        """Validate initialization parameters."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
    
    def _validate_texts(self, texts: Union[str, List[str], pd.Series]) -> List[str]:
        """
        Validate and normalize text input.
        
        Args:
            texts: Input texts to validate
            
        Returns:
            List of validated text strings
            
        Raises:
            ValueError: If texts is invalid
        """
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif isinstance(texts, list):
            texts = texts
        else:
            raise ValueError("texts must be a string, list of strings, or pandas Series")
        
        # Check for empty or None texts
        if not texts:
            raise ValueError("texts cannot be empty")
        
        # Convert None values to empty strings and validate
        validated_texts = []
        for i, text in enumerate(texts):
            if text is None:
                logger.warning(f"Found None value at index {i}, converting to empty string")
                validated_texts.append("")
            elif not isinstance(text, str):
                raise ValueError(f"All texts must be strings, found {type(text)} at index {i}")
            else:
                validated_texts.append(text)
        
        return validated_texts
    
    def encode_texts(self, 
                    texts: Union[str, List[str], pd.Series],
                    padding: Union[bool, str] = True,
                    truncation: bool = True,
                    return_tensors: Optional[str] = 'pt',
                    return_attention_mask: bool = True,
                    return_token_type_ids: bool = False,
                    add_special_tokens: bool = True) -> Dict[str, Any]:
        """
        Encode texts for transformer model input.
        
        Args:
            texts: Input texts to encode
            padding: Whether to pad sequences (default: True)
            truncation: Whether to truncate sequences (default: True)
            return_tensors: Format of returned tensors ('pt', 'tf', 'np', or None)
            return_attention_mask: Whether to return attention masks
            return_token_type_ids: Whether to return token type IDs
            add_special_tokens: Whether to add special tokens ([CLS], [SEP])
            
        Returns:
            Dictionary containing encoded texts with keys like 'input_ids', 'attention_mask'
            
        Raises:
            ValueError: If input validation fails
        """
        validated_texts = self._validate_texts(texts)
        
        try:
            encoded = self.tokenizer(
                validated_texts,
                padding=padding,
                truncation=truncation,
                max_length=self.max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
                add_special_tokens=add_special_tokens
            )
            
            logger.debug(f"Encoded {len(validated_texts)} texts with max_length={self.max_length}")
            return encoded
            
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts: {str(e)}")
    
    def decode_texts(self, 
                    token_ids: Union[torch.Tensor, np.ndarray, List[int], List[List[int]]],
                    skip_special_tokens: bool = True,
                    clean_up_tokenization_spaces: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces
            
        Returns:
            Decoded text(s)
            
        Raises:
            ValueError: If token_ids format is invalid
        """
        try:
            # Handle different input formats
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            elif isinstance(token_ids, np.ndarray):
                token_ids = token_ids.tolist()
            
            # Check if it's a single sequence or batch
            if isinstance(token_ids[0], int):
                # Single sequence
                return self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces
                )
            else:
                # Batch of sequences
                return self.tokenizer.batch_decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces
                )
                
        except Exception as e:
            raise ValueError(f"Failed to decode token IDs: {str(e)}")
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Get the special tokens used by the tokenizer."""
        return {
            'cls_token': self.tokenizer.cls_token,
            'sep_token': self.tokenizer.sep_token,
            'pad_token': self.tokenizer.pad_token,
            'unk_token': self.tokenizer.unk_token,
            'mask_token': getattr(self.tokenizer, 'mask_token', None)
        }
    
    def batch_encode_texts(self, 
                          texts: Union[List[str], pd.Series],
                          batch_size: int = 32,
                          **kwargs) -> List[Dict[str, Any]]:
        """
        Encode texts in batches for memory efficiency.
        
        Args:
            texts: Input texts to encode
            batch_size: Number of texts to process at once
            **kwargs: Additional arguments for encode_texts
            
        Returns:
            List of encoded batches
        """
        validated_texts = self._validate_texts(texts)
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        batches = []
        for i in range(0, len(validated_texts), batch_size):
            batch_texts = validated_texts[i:i + batch_size]
            batch_encoded = self.encode_texts(batch_texts, **kwargs)
            batches.append(batch_encoded)
        
        logger.info(f"Processed {len(validated_texts)} texts in {len(batches)} batches")
        return batches
    
    def get_text_statistics(self, texts: Union[List[str], pd.Series]) -> Dict[str, Union[int, float]]:
        """
        Get statistics about tokenized texts.
        
        Args:
            texts: Input texts to analyze
            
        Returns:
            Dictionary with statistics about token lengths
        """
        validated_texts = self._validate_texts(texts)
        
        # Tokenize to get lengths
        token_lengths = []
        for text in validated_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            token_lengths.append(len(tokens))
        
        return {
            'num_texts': len(validated_texts),
            'avg_token_length': np.mean(token_lengths),
            'min_token_length': np.min(token_lengths),
            'max_token_length': np.max(token_lengths),
            'std_token_length': np.std(token_lengths),
            'texts_exceeding_max_length': sum(1 for length in token_lengths if length > self.max_length)
        }
    
    def __repr__(self) -> str:
        """String representation of the processor."""
        return (f"TransformerTextProcessor(model_name='{self.model_name}', "
                f"max_length={self.max_length}, "
                f"vocab_size={self.get_vocab_size()})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"TransformerTextProcessor using {self.model_name}"

    @classmethod
    def get_recommended_models(cls, max_sequence_length: int = 512) -> Dict[str, Dict[str, Union[str, int]]]:
        """
        Get recommended models based on maximum sequence length requirements.
        
        Args:
            max_sequence_length: Required maximum sequence length
            
        Returns:
            Dictionary of recommended models with their specifications
        """
        models = {
            # Standard BERT models (512 tokens)
            'bert-base-uncased': {
                'max_length': 512,
                'description': 'Standard BERT, good baseline performance',
                'best_for': 'Short to medium reviews (< 512 tokens)'
            },
            'bert-large-uncased': {
                'max_length': 512,
                'description': 'Larger BERT, better performance, same length limit',
                'best_for': 'Short to medium reviews, when accuracy is important'
            },
            'roberta-base': {
                'max_length': 512,
                'description': 'RoBERTa, often outperforms BERT',
                'best_for': 'Short to medium reviews, robust performance'
            },
            'roberta-large': {
                'max_length': 512,
                'description': 'Large RoBERTa, excellent performance',
                'best_for': 'Short to medium reviews, highest accuracy'
            },
            
            # Medium length models (1024 tokens)
            'microsoft/deberta-base': {
                'max_length': 512,
                'description': 'DeBERTa, state-of-the-art performance',
                'best_for': 'Better than BERT, same length limit'
            },
            'microsoft/deberta-v2-xlarge': {
                'max_length': 1024,
                'description': 'DeBERTa v2, extended context',
                'best_for': 'Medium length reviews, excellent performance'
            },
            
            # Long context models (4096+ tokens)
            'allenai/longformer-base-4096': {
                'max_length': 4096,
                'description': 'Longformer, designed for long documents',
                'best_for': 'Long reviews, full context understanding'
            },
            'google/bigbird-roberta-base': {
                'max_length': 4096,
                'description': 'BigBird, sparse attention for long sequences',
                'best_for': 'Very long reviews, memory efficient'
            },
            'allenai/led-base-16384': {
                'max_length': 16384,
                'description': 'LED, extremely long context',
                'best_for': 'Very long documents, research use'
            }
        }
        
        # Filter models that can handle the required sequence length
        suitable_models = {
            name: specs for name, specs in models.items() 
            if specs['max_length'] >= max_sequence_length
        }
        
        return suitable_models
    
    @classmethod
    def print_model_recommendations(cls, max_sequence_length: int = 512) -> None:
        """
        Print model recommendations based on sequence length requirements.
        
        Args:
            max_sequence_length: Required maximum sequence length
        """
        models = cls.get_recommended_models(max_sequence_length)
        
        print(f"Models suitable for max sequence length {max_sequence_length}:")
        print("=" * 70)
        
        for name, specs in models.items():
            print(f"\n{name}")
            print(f"  Max Length: {specs['max_length']} tokens")
            print(f"  Description: {specs['description']}")
            print(f"  Best For: {specs['best_for']}")
    
    def analyze_sequence_lengths(self, texts: Union[List[str], pd.Series]) -> Dict[str, Union[int, float, List[str]]]:
        """
        Analyze sequence lengths and provide model recommendations.
        
        Args:
            texts: Input texts to analyze
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        validated_texts = self._validate_texts(texts)
        
        # Get token lengths
        token_lengths = []
        long_texts = []
        
        for i, text in enumerate(validated_texts):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            token_lengths.append(len(tokens))
            
            if len(tokens) > self.max_length:
                long_texts.append({
                    'index': i,
                    'length': len(tokens),
                    'text_preview': text[:100] + "..." if len(text) > 100 else text
                })
        
        # Calculate statistics
        max_length_needed = max(token_lengths) if token_lengths else 0
        texts_exceeding_limit = sum(1 for length in token_lengths if length > self.max_length)
        
        # Get model recommendations
        recommended_models = self.get_recommended_models(max_length_needed)
        
        analysis = {
            'num_texts': len(validated_texts),
            'avg_token_length': np.mean(token_lengths),
            'min_token_length': min(token_lengths) if token_lengths else 0,
            'max_token_length': max_length_needed,
            'std_token_length': np.std(token_lengths),
            'texts_exceeding_current_limit': texts_exceeding_limit,
            'percentage_exceeding_limit': (texts_exceeding_limit / len(validated_texts)) * 100 if validated_texts else 0,
            'max_length_needed': max_length_needed,
            'current_model_max_length': self.max_length,
            'recommended_models': list(recommended_models.keys())[:3],  # Top 3 recommendations
            'long_text_examples': long_texts[:5]  # First 5 examples
        }
        
        return analysis
    
    def print_sequence_analysis(self, texts: Union[List[str], pd.Series]) -> None:
        """
        Print a detailed analysis of sequence lengths with recommendations.
        
        Args:
            texts: Input texts to analyze
        """
        analysis = self.analyze_sequence_lengths(texts)
        
        print("SEQUENCE LENGTH ANALYSIS")
        print("=" * 50)
        print(f"Total texts: {analysis['num_texts']}")
        print(f"Average token length: {analysis['avg_token_length']:.1f}")
        print(f"Max token length: {analysis['max_token_length']}")
        print(f"Current model limit: {analysis['current_model_max_length']}")
        print(f"Texts exceeding limit: {analysis['texts_exceeding_current_limit']} ({analysis['percentage_exceeding_limit']:.1f}%)")
        
        if analysis['texts_exceeding_current_limit'] > 0:
            print(f"\n⚠️  WARNING: {analysis['texts_exceeding_current_limit']} texts exceed the current model's limit!")
            print(f"Max length needed: {analysis['max_length_needed']} tokens")
            
            print("\nRECOMMENDED MODELS:")
            for model_name in analysis['recommended_models']:
                models = self.get_recommended_models(analysis['max_length_needed'])
                if model_name in models:
                    model_info = models[model_name]
                    print(f"  • {model_name} (max: {model_info['max_length']} tokens)")
                    print(f"    {model_info['description']}")
            
            if analysis['long_text_examples']:
                print(f"\nEXAMPLES OF LONG TEXTS:")
                for example in analysis['long_text_examples']:
                    print(f"  Text {example['index']}: {example['length']} tokens")
                    print(f"    Preview: {example['text_preview']}")
        else:
            print("\n✅ All texts fit within the current model's limit.")
