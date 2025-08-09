"""
Baseline model for text classification.

This module implements a simple baseline model using TF-IDF
vectorization and logistic regression for binary text classification.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class BaselineModel:
    """
    Baseline text classification model using TF-IDF + Logistic Regression.
    
    This model serves as a reference point for more advanced models.
    It includes evaluation metrics and model persistence.
    """
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 random_state: int = 42,
                 test_size: float = 0.2):
        """
        Initialize the baseline model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for testing
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.test_size = test_size
        
        # Initialize the pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )),
            ('classifier', LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                solver='liblinear'
            ))
        ])
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Evaluation results
        self.evaluation_results = {}
        
    def prepare_data(self, texts: pd.Series, labels: pd.Series) -> None:
        """
        Prepare and split the data for training.
        
        Args:
            texts: Series of text documents
            labels: Series of binary labels
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=labels
        )
        
        print(f"Data split complete:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        print(f"  Class distribution (train): {self.y_train.value_counts().to_dict()}")
        print(f"  Class distribution (test): {self.y_test.value_counts().to_dict()}")
    
    def train(self) -> Dict[str, float]:
        """
        Train the baseline model.
        
        Returns:
            Dict containing training metrics
        """
        print("Training baseline model...")
        
        # Train the pipeline
        self.pipeline.fit(self.X_train, self.y_train)
        
        # Get training predictions
        y_train_pred = self.pipeline.predict(self.X_train)
        y_train_proba = self.pipeline.predict_proba(self.X_train)[:, 1]
        
        # Calculate training metrics
        train_metrics = {
            'accuracy': accuracy_score(self.y_train, y_train_pred),
            'precision': precision_score(self.y_train, y_train_pred),
            'recall': recall_score(self.y_train, y_train_pred),
            'f1': f1_score(self.y_train, y_train_pred),
            'roc_auc': roc_auc_score(self.y_train, y_train_proba)
        }
        
        print("Training completed!")
        print(f"Training metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return train_metrics
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Dict containing comprehensive evaluation results
        """
        print("Evaluating baseline model...")
        
        # Get predictions
        y_pred = self.pipeline.predict(self.X_test)
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(self.y_test, y_proba)
        
        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(self.y_test, y_proba)
        
        # Store results
        self.evaluation_results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_proba,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        }
        
        print("Evaluation completed!")
        print(f"Test metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return self.evaluation_results
    
    def cross_validate(self, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the training data.
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dict containing cross-validation results
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.pipeline, self.X_train, self.y_train, 
            cv=cv_folds, scoring='f1'
        )
        
        cv_results = {
            'mean_f1': cv_scores.mean(),
            'std_f1': cv_scores.std(),
            'f1_scores': cv_scores.tolist()
        }
        
        print(f"Cross-validation results:")
        print(f"  Mean F1: {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1']:.4f})")
        
        return cv_results
    
    def predict(self, texts: pd.Series) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            texts: Series of text documents
            
        Returns:
            Array of predicted labels
        """
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """
        Get prediction probabilities on new data.
        
        Args:
            texts: Series of text documents
            
        Returns:
            Array of prediction probabilities
        """
        return self.pipeline.predict_proba(texts)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'evaluation_results': self.evaluation_results,
                'model_params': {
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'random_state': self.random_state,
                    'test_size': self.test_size
                }
            }, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaselineModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded BaselineModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance with saved parameters
        model = cls(**model_data['model_params'])
        
        # Restore pipeline and results
        model.pipeline = model_data['pipeline']
        model.evaluation_results = model_data.get('evaluation_results', {})
        
        print(f"Model loaded from: {filepath}")
        return model
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get the most important features (words) for classification.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and their coefficients
        """
        # Get feature names from TF-IDF
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        
        # Get coefficients from logistic regression
        coefficients = self.pipeline.named_steps['classifier'].coef_[0]
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient value
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        return feature_importance.head(top_n)
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"BaselineModel(max_features={self.max_features}, ngram_range={self.ngram_range})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return (f"BaselineModel(max_features={self.max_features}, "
                f"ngram_range={self.ngram_range}, "
                f"random_state={self.random_state}, "
                f"test_size={self.test_size})") 