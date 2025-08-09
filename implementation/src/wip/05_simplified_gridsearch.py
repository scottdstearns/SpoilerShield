#!/usr/bin/env python3
"""
SpoilerShield: Simplified Grid Search for A/B Testing
====================================================

This script implements a simplified grid search for rapid A/B comparison with genetic algorithms.
Reduced parameter space for quick iteration and testing.

Author: SpoilerShield Development Team
Date: 2025-08-07
"""

import sys
import warnings
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import os

# Machine Learning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    make_scorer
)

# Transformers (conditional import)
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
        set_seed as transformers_set_seed
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Only LogisticRegression grid search will run.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
src_path = Path(__file__).parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader


def set_all_seeds(seed: int):
    """
    Set seeds for all random number generators for full reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random
    import random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch (if available)
    if TRANSFORMERS_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        
        # Additional PyTorch reproducibility settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Use transformers seed function
        transformers_set_seed(seed)
    
    print(f"🔒 All random seeds set to: {seed}")


class SimplifiedGridSearchOptimizer:
    """
    Simplified hyperparameter grid search for rapid A/B testing with genetic algorithms.
    
    Reduced parameter space for quick iteration:
    - LogReg: 6 combinations (~10 minutes)
    - RoBERTa: 4 combinations (~20 minutes)
    - Total: 30 minutes
    """
    
    def __init__(self, config: EnvConfig, random_state: int = 42):
        """Initialize the SimplifiedGridSearchOptimizer."""
        self.config = config
        self.random_state = random_state
        self.results = {}
        
        # Set all random seeds for reproducibility
        set_all_seeds(random_state)
        
        # Check GPU availability
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")
        
        print("⚡ SPOILERSHIELD: SIMPLIFIED GRID SEARCH (A/B Testing)")
        print("=" * 60)
    
    def prepare_data(self, df_reviews: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for grid search experiments."""
        print("\n📝 PREPARING DATA")
        print("-" * 30)
        
        # Extract text and labels
        texts = df_reviews['review_text'].values
        labels = df_reviews['is_spoiler'].values
        
        # Re-seed before data split for consistency
        set_all_seeds(self.random_state)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        print(f"✅ Training samples: {len(X_train):,}")
        print(f"✅ Test samples: {len(X_test):,}")
        print(f"✅ Class distribution (train): {np.bincount(y_train)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_simplified_logistic_grid(self) -> Dict[str, List]:
        """Define simplified parameter grid for LogisticRegression."""
        return {
            # Core parameters only (6 combinations)
            'tfidf__max_features': [10000, 20000],              # 2 values
            'classifier__C': [1.0, 5.0],                       # 2 values  
            'classifier__penalty': ['l2'],                      # 1 value (simplified)
            
            # Fixed optimal parameters
            'tfidf__ngram_range': [(1, 2)],                     # Fixed at proven winner
            'tfidf__min_df': [2],
            'tfidf__max_df': [0.95],
            'tfidf__sublinear_tf': [True],
            'classifier__class_weight': ['balanced'],
            'classifier__max_iter': [2000],
        }
    
    def get_simplified_roberta_grid(self) -> Dict[str, List]:
        """Define simplified parameter grid for RoBERTa."""
        return {
            # Core parameters only (4 combinations)
            'learning_rate': [3e-5, 5e-5],                     # 2 values
            'max_length': [256, 512],                          # 2 values
            
            # Fixed parameters for speed
            'model_name': ['roberta-base'],                     # Fixed (skip large)
            'num_train_epochs': [3],                           # Fixed
            'per_device_train_batch_size': [16],               # Fixed
            'weight_decay': [0.01],
            'warmup_ratio': [0.1],
            'dropout_rate': [0.1],
        }
    
    def optimize_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Perform simplified grid search for LogisticRegression."""
        print("\n⚡ SIMPLIFIED LOGISTIC REGRESSION GRID SEARCH")
        print("=" * 50)
        
        start_time = time.time()
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode')),
            ('classifier', LogisticRegression(random_state=self.random_state, solver='lbfgs', n_jobs=1))
        ])
        
        # Get simplified parameter grid
        param_grid = self.get_simplified_logistic_grid()
        
        # Calculate combinations
        total_combinations = 2 * 2 * 1  # max_features × C × penalty
        print(f"📊 Parameter combinations: {total_combinations}")
        print(f"📊 With 3-fold CV: {total_combinations * 3} model fits")
        print(f"📊 Estimated time: ~8-10 minutes")
        
        # Set up cross-validation
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        f1_scorer = make_scorer(f1_score)
        
        # Perform grid search
        print("\n🚀 Starting simplified grid search...")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=f1_scorer,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get results
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        test_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        optimization_time = time.time() - start_time
        
        results = {
            'model_type': 'LogisticRegression_Simplified',
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_metrics': test_metrics,
            'optimization_time': optimization_time,
            'total_combinations': total_combinations,
            'method': 'GridSearchCV_Simplified'
        }
        
        print(f"\n✅ Simplified grid search completed in {optimization_time:.1f} seconds")
        print(f"🏆 Best CV F1 Score: {grid_search.best_score_:.4f}")
        print(f"🎯 Test F1 Score: {test_metrics['f1']:.4f}")
        print(f"📊 Best parameters: {grid_search.best_params_}")
        
        return results
    
    def optimize_roberta_simplified(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Perform simplified grid search for RoBERTa."""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available. Skipping RoBERTa optimization.")
            return {'error': 'Transformers not available'}
        
        print("\n⚡ SIMPLIFIED ROBERTA GRID SEARCH")
        print("=" * 50)
        
        start_time = time.time()
        param_grid = self.get_simplified_roberta_grid()
        
        # Calculate combinations
        total_combinations = 2 * 2  # learning_rate × max_length
        print(f"📊 Parameter combinations: {total_combinations}")
        print(f"📊 With 3-fold CV: {total_combinations * 3} model fits")
        print(f"📊 Estimated time: ~15-20 minutes")
        
        # Test each combination
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        best_score = 0.0
        best_params = None
        all_results = []
        
        print("\n🚀 Starting RoBERTa simplified search...")
        
        # Generate parameter combinations
        learning_rates = param_grid['learning_rate']
        max_lengths = param_grid['max_length']
        
        for lr in learning_rates:
            for max_len in max_lengths:
                params = {
                    'model_name': 'roberta-base',
                    'learning_rate': lr,
                    'max_length': max_len,
                    'num_train_epochs': 3,
                    'per_device_train_batch_size': 16,
                    'weight_decay': 0.01,
                    'warmup_ratio': 0.1,
                    'dropout_rate': 0.1
                }
                
                print(f"\n🔄 Testing: LR={lr}, MaxLen={max_len}")
                
                # Perform 3-fold CV for this configuration
                cv_scores = []
                for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
                    print(f"  📊 Fold {fold+1}/3...")
                    
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    try:
                        fold_score = self._train_roberta_fold(
                            params, X_fold_train, y_fold_train, X_fold_val, y_fold_val
                        )
                        cv_scores.append(fold_score)
                        print(f"    F1: {fold_score:.4f}")
                    except Exception as e:
                        print(f"    ❌ Fold failed: {str(e)}")
                        cv_scores.append(0.0)
                
                mean_cv_score = np.mean(cv_scores)
                print(f"  📈 CV F1: {mean_cv_score:.4f}")
                
                all_results.append({
                    'params': params,
                    'cv_score': mean_cv_score,
                    'cv_scores': cv_scores
                })
                
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_params = params.copy()
                    print(f"  🏆 New best: {best_score:.4f}")
        
        # Train final model with best parameters
        print(f"\n🎯 Training final model with best params...")
        try:
            final_score = self._train_roberta_fold(best_params, X_train, y_train, X_test, y_test)
            optimization_time = time.time() - start_time
            
            results = {
                'model_type': 'RoBERTa_Simplified',
                'best_params': best_params,
                'best_cv_score': best_score,
                'test_metrics': {'f1': final_score},
                'optimization_time': optimization_time,
                'total_combinations': total_combinations,
                'method': 'GridSearchCV_Simplified',
                'all_results': all_results
            }
            
            print(f"\n✅ RoBERTa simplified search completed in {optimization_time:.1f} seconds")
            print(f"🏆 Best CV F1: {best_score:.4f}")
            print(f"🎯 Test F1: {final_score:.4f}")
            
        except Exception as e:
            print(f"❌ Final training failed: {str(e)}")
            results = {'error': str(e), 'method': 'GridSearchCV_Simplified'}
        
        return results
    
    def _train_roberta_fold(self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train a single RoBERTa fold and return F1 score."""
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            params['model_name'], 
            num_labels=2,
            hidden_dropout_prob=params['dropout_rate']
        )
        
        # Create datasets
        train_dataset = self._create_dataset(X_train, y_train, tokenizer, params['max_length'])
        val_dataset = self._create_dataset(X_val, y_val, tokenizer, params['max_length'])
        
        # Training arguments with reproducibility settings
        training_args = TrainingArguments(
            output_dir=self.config.output_dir / 'temp_roberta_simplified',
            num_train_epochs=params['num_train_epochs'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=params['per_device_train_batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            warmup_ratio=params['warmup_ratio'],
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="no",  # Don't save for simplified search
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            # Reproducibility settings
            seed=self.random_state,
            data_seed=self.random_state,
            dataloader_num_workers=0,  # Ensures deterministic data loading
        )
        
        # Compute metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'f1': f1_score(labels, predictions)}
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        
        # Cleanup
        del model, trainer, train_dataset, val_dataset
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return eval_results.get('eval_f1', 0.0)
    
    def _create_dataset(self, texts: np.ndarray, labels: np.ndarray, 
                       tokenizer: Any, max_length: int) -> Dataset:
        """Create HuggingFace Dataset."""
        encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length, return_tensors='pt'
        )
        
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels.astype(int)
        })
    
    def save_results(self, logreg_results: Dict[str, Any], roberta_results: Dict[str, Any]):
        """Save simplified results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'method': 'Simplified_GridSearch',
            'logistic_regression': logreg_results,
            'roberta': roberta_results,
            'purpose': 'A/B_Testing_Baseline_for_GA_Comparison'
        }
        
        # Save results
        results_file = self.config.output_dir / f"simplified_gridsearch_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Simplified results saved: {results_file}")
        return results_file
    
    def run_simplified_optimization(self, df_reviews: pd.DataFrame) -> Dict[str, Any]:
        """Run the complete simplified optimization."""
        print("⚡ STARTING SIMPLIFIED GRID SEARCH (A/B BASELINE)")
        print("=" * 60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df_reviews)
        
        # Run LogReg optimization
        print("\n" + "="*60)
        logreg_results = self.optimize_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Run RoBERTa optimization
        print("\n" + "="*60)
        roberta_results = self.optimize_roberta_simplified(X_train, y_train, X_test, y_test)
        
        # Save results
        results_file = self.save_results(logreg_results, roberta_results)
        
        print("\n⚡ SIMPLIFIED GRID SEARCH COMPLETE!")
        print("=" * 60)
        print(f"📁 Results saved to: {results_file}")
        print(f"🎯 Ready for GA A/B comparison!")
        
        return {
            'logistic_regression': logreg_results,
            'roberta': roberta_results,
            'results_file': results_file
        }


def main():
    """Main execution function."""
    print("⚡ SPOILERSHIELD: SIMPLIFIED GRID SEARCH")
    print("=" * 60)
    
    # Initialize
    config = EnvConfig()
    
    # Load data
    print("\n📥 LOADING DATA")
    print("-" * 30)
    
    data_loader = DataLoader(
        movie_reviews_path=config.get_data_path('train_reviews.json'),
        movie_details_path=config.get_data_path('IMDB_movie_details.json')
    )
    
    df_reviews = data_loader.load_imdb_movie_reviews()
    print(f"✅ Loaded {len(df_reviews):,} reviews")
    
    # Run optimization
    optimizer = SimplifiedGridSearchOptimizer(config)
    results = optimizer.run_simplified_optimization(df_reviews)
    
    print(f"\n📊 SUMMARY:")
    if 'error' not in results['logistic_regression']:
        lr_f1 = results['logistic_regression']['test_metrics']['f1']
        print(f"  LogReg F1: {lr_f1:.4f}")
    
    if 'error' not in results['roberta']:
        roberta_f1 = results['roberta']['test_metrics']['f1']
        print(f"  RoBERTa F1: {roberta_f1:.4f}")


if __name__ == "__main__":
    main()
