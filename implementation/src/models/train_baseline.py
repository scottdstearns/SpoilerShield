"""
Baseline model training script.

This script demonstrates the complete workflow for training, evaluating,
and saving a baseline text classification model.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from models.baseline_model import BaselineModel
from evaluation.model_evaluator import ModelEvaluator
from eda.data_loader import DataLoader
from utils.env_config import EnvConfig


def train_baseline_model(data_path: str = None, 
                        output_dir: str = None,
                        model_name: str = "Baseline_TFIDF_LogReg") -> BaselineModel:
    """
    Train a baseline model with comprehensive evaluation.
    
    Args:
        data_path: Path to the data directory (optional, uses config if None)
        output_dir: Directory to save outputs (optional, uses config if None)
        model_name: Name for the model
        
    Returns:
        Trained BaselineModel instance
    """
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Initialize environment configuration
    config = EnvConfig()
    print(f"Environment: {config.env}")
    print(f"Root directory: {config.root_dir}")
    
    # Set up paths
    if data_path is None:
        data_path = str(config.data_dir)
    if output_dir is None:
        output_dir = str(config.output_dir)
    
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    data_loader = DataLoader(
        os.path.join(data_path, 'train_reviews.json'),
        os.path.join(data_path, 'IMDB_movie_details.json')
    )
    
    df_reviews = data_loader.load_imdb_movie_reviews()
    print(f"Loaded {len(df_reviews)} reviews")
    
    # Check for missing values
    print(f"Missing values in review_text: {df_reviews['review_text'].isnull().sum()}")
    print(f"Missing values in is_spoiler: {df_reviews['is_spoiler'].isnull().sum()}")
    
    # Remove any rows with missing values
    df_reviews = df_reviews.dropna(subset=['review_text', 'is_spoiler'])
    print(f"After removing missing values: {len(df_reviews)} reviews")
    
    # Initialize the baseline model
    print("\nInitializing baseline model...")
    baseline_model = BaselineModel(
        max_features=10000,
        ngram_range=(1, 2),
        random_state=42,
        test_size=0.2
    )
    
    # Prepare data
    print("\nPreparing data...")
    baseline_model.prepare_data(
        texts=df_reviews['review_text'],
        labels=df_reviews['is_spoiler']
    )
    
    # Train the model
    print("\nTraining model...")
    train_metrics = baseline_model.train()
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluation_results = baseline_model.evaluate()
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    cv_results = baseline_model.cross_validate(cv_folds=5)
    
    # Initialize model evaluator
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Evaluate with the evaluator for comprehensive analysis
    evaluator.evaluate_model(
        model_name=model_name,
        y_true=baseline_model.y_test,
        y_pred=evaluation_results['predictions'],
        y_proba=evaluation_results['probabilities'],
        save_plots=True
    )
    
    # Generate comparison plots
    print("\nGenerating evaluation plots...")
    evaluator.plot_confusion_matrices(save_plot=True)
    evaluator.plot_roc_curves(save_plot=True)
    evaluator.plot_precision_recall_curves(save_plot=True)
    evaluator.plot_metrics_comparison(save_plot=True)
    
    # Save evaluation report
    evaluator.save_evaluation_report(f'{model_name}_evaluation_report.txt')
    
    # Get feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = baseline_model.get_feature_importance(top_n=20)
    print("\nTop 20 most important features:")
    print(feature_importance)
    
    # Save feature importance
    feature_importance_path = os.path.join(output_dir, f'{model_name}_feature_importance.csv')
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"Feature importance saved to: {feature_importance_path}")
    
    # Save the trained model
    model_path = os.path.join(output_dir, f'{model_name}.pkl')
    baseline_model.save_model(model_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Training samples: {len(baseline_model.X_train)}")
    print(f"Test samples: {len(baseline_model.X_test)}")
    print(f"Max features: {baseline_model.max_features}")
    print(f"N-gram range: {baseline_model.ngram_range}")
    print(f"Model saved to: {model_path}")
    print(f"Outputs saved to: {output_dir}")
    
    print("\nFinal Test Metrics:")
    for metric, value in evaluation_results['metrics'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    return baseline_model


def load_and_evaluate_model(model_path: str, 
                           test_texts: pd.Series = None,
                           test_labels: pd.Series = None) -> BaselineModel:
    """
    Load a saved model and evaluate it on new data.
    
    Args:
        model_path: Path to the saved model
        test_texts: Test texts for evaluation (optional)
        test_labels: Test labels for evaluation (optional)
        
    Returns:
        Loaded BaselineModel instance
    """
    print(f"Loading model from: {model_path}")
    
    # Load the model
    model = BaselineModel.load_model(model_path)
    
    # If test data is provided, evaluate
    if test_texts is not None and test_labels is not None:
        print("Evaluating loaded model on new data...")
        
        # Make predictions
        y_pred = model.predict(test_texts)
        y_proba = model.predict_proba(test_texts)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(test_labels, y_pred),
            'precision': precision_score(test_labels, y_pred),
            'recall': recall_score(test_labels, y_pred),
            'f1': f1_score(test_labels, y_pred)
        }
        
        print("Evaluation results:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    return model


if __name__ == "__main__":
    # Train the baseline model
    model = train_baseline_model()
    
    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING COMPLETED")
    print("=" * 60)
    print("Next steps:")
    print("1. Review the evaluation plots and reports")
    print("2. Analyze feature importance for insights")
    print("3. Use this as a baseline for more advanced models")
    print("4. Experiment with different hyperparameters")
    print("=" * 60) 