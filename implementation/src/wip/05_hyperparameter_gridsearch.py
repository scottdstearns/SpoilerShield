#!/usr/bin/env python3
"""
SpoilerShield: Hyperparameter Grid Search Optimization
======================================================

This script implements comprehensive grid search for hyperparameter optimization using:
1. LogisticRegression + TF-IDF pipeline with stratified cross-validation
2. RoBERTa transformer models (base vs large) with systematic parameter search
3. Performance tracking and comparison framework for A/B testing with genetic algorithms

Author: SpoilerShield Development Team
Date: 2025-08-07
"""

import sys
import warnings
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        TrainingArguments, Trainer, EarlyStoppingCallback
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Only LogisticRegression grid search will run.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path for our custom modules
src_path = Path(__file__).parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader
from evaluation.model_evaluator import ModelEvaluator


class GridSearchOptimizer:
    """
    Comprehensive hyperparameter grid search optimizer.
    
    Implements systematic parameter search for:
    1. Traditional ML: LogisticRegression + TF-IDF
    2. Transformers: RoBERTa base/large models
    """
    
    def __init__(self, config: EnvConfig, random_state: int = 42):
        """
        Initialize the GridSearchOptimizer.
        
        Args:
            config: Environment configuration
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.results = {}
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(output_dir=str(config.output_dir))
        
        # Check GPU availability for transformers
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")
        
        print("🔍 SPOILERSHIELD: HYPERPARAMETER GRID SEARCH OPTIMIZATION")
        print("=" * 70)
    
    def prepare_data(self, df_reviews: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for grid search experiments.
        
        Args:
            df_reviews: DataFrame with review data
            
        Returns:
            X_train, X_test, y_train, y_test splits
        """
        print("\n📝 PREPARING DATA FOR GRID SEARCH")
        print("-" * 50)
        
        # Extract text and labels
        texts = df_reviews['review_text'].values
        labels = df_reviews['is_spoiler'].values
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        print(f"✅ Training samples: {len(X_train):,}")
        print(f"✅ Test samples: {len(X_test):,}")
        print(f"✅ Class distribution (train): {np.bincount(y_train)}")
        print(f"✅ Class distribution (test): {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_logistic_regression_pipeline(self) -> Pipeline:
        """
        Create LogisticRegression pipeline with TF-IDF.
        
        Returns:
            Sklearn pipeline
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )),
            ('classifier', LogisticRegression(
                random_state=self.random_state,
                solver='saga',  # Supports all penalties
                n_jobs=1  # Controlled parallelism at GridSearch level
            ))
        ])
        
        return pipeline
    
    def get_logistic_regression_param_grid(self) -> Dict[str, List]:
        """
        Define parameter grid for LogisticRegression + TF-IDF.
        
        Returns:
            Parameter grid dictionary
        """
        param_grid = {
            # TF-IDF Parameters
            'tfidf__max_features': [10000, 20000],              # Vocabulary size
            'tfidf__ngram_range': [(1, 2), (1, 3)],            # N-gram complexity  
            
            # LogisticRegression Parameters
            'classifier__C': [1.0, 2.0, 5.0],                  # Regularization strength
            'classifier__penalty': ['l2', 'elasticnet'],        # Regularization type
            
            # Fixed optimal parameters from class imbalance analysis
            'tfidf__min_df': [2],                               # Minimal noise filtering
            'tfidf__max_df': [0.95],                            # Standard common word filtering
            'tfidf__sublinear_tf': [True],                      # Log-scale TF (generally better)
            'classifier__class_weight': ['balanced'],            # Proven winner from imbalance analysis
            'classifier__max_iter': [2000],                     # Ensure convergence
            'classifier__l1_ratio': [0.5],                      # Balanced ElasticNet (only used when penalty='elasticnet')
        }
        
        return param_grid
    
    def optimize_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Perform grid search optimization for LogisticRegression.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Optimization results dictionary
        """
        print("\n🔍 LOGISTIC REGRESSION GRID SEARCH")
        print("=" * 50)
        
        start_time = time.time()
        
        # Create pipeline and parameter grid
        pipeline = self.create_logistic_regression_pipeline()
        param_grid = self.get_logistic_regression_param_grid()
        
        # Calculate total combinations
        total_combinations = 1
        for param_list in param_grid.values():
            total_combinations *= len(param_list)
        
        print(f"📊 Parameter grid combinations: {total_combinations}")
        print(f"📊 With 3-fold CV: {total_combinations * 3} model fits")
        print(f"📊 Estimated time: {total_combinations * 3 * 30 / 60:.1f} minutes")
        
        # Set up stratified cross-validation
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        # Create custom scoring function
        f1_scorer = make_scorer(f1_score)
        
        # Perform grid search
        print("\n🚀 Starting grid search...")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=f1_scorer,
            n_jobs=-1,  # Use all available cores
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model and predictions
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
            'model_type': 'LogisticRegression',
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_metrics': test_metrics,
            'optimization_time': optimization_time,
            'total_combinations': total_combinations,
            'cv_results': grid_search.cv_results_,
            'best_model': best_model,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"\n✅ Grid search completed in {optimization_time:.2f} seconds")
        print(f"🏆 Best CV F1 Score: {grid_search.best_score_:.4f}")
        print(f"🎯 Test F1 Score: {test_metrics['f1']:.4f}")
        print(f"📊 Best parameters: {grid_search.best_params_}")
        
        return results
    
    def get_roberta_param_grid(self) -> Dict[str, List]:
        """
        Define parameter grid for RoBERTa models.
        
        Returns:
            Parameter grid dictionary
        """
        param_grid = {
            # Model Architecture
            'model_name': ['roberta-base', 'roberta-large'],
            
            # Training Parameters
            'learning_rate': [3e-5, 5e-5],
            'num_train_epochs': [3, 4],
            'max_length': [256, 512],
            
            # Fixed parameters (from analysis)
            'per_device_train_batch_size': [12],  # Reduced for roberta-large memory
            'weight_decay': [0.01],
            'warmup_ratio': [0.1],
            'dropout_rate': [0.1],
        }
        
        return param_grid
    
    def create_transformer_dataset(self, texts: np.ndarray, labels: np.ndarray, 
                                 tokenizer: Any, max_length: int) -> Dataset:
        """
        Create HuggingFace Dataset for transformer training.
        
        Args:
            texts: Text data
            labels: Label data
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
            
        Returns:
            HuggingFace Dataset
        """
        # Tokenize texts
        encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels.astype(int)
        })
        
        return dataset
    
    def train_single_roberta_config(self, params: Dict[str, Any],
                                   X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Train a single RoBERTa configuration.
        
        Args:
            params: Parameter configuration
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Metrics dictionary
        """
        start_time = time.time()
        
        # Load model and tokenizer
        model_name = params['model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            hidden_dropout_prob=params['dropout_rate']
        )
        
        # Create datasets
        train_dataset = self.create_transformer_dataset(
            X_train, y_train, tokenizer, params['max_length']
        )
        val_dataset = self.create_transformer_dataset(
            X_val, y_val, tokenizer, params['max_length']
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir / 'temp_roberta_training',
            num_train_epochs=params['num_train_epochs'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=params['per_device_train_batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            warmup_ratio=params['warmup_ratio'],
            logging_dir=self.config.output_dir / 'temp_roberta_logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            f1 = f1_score(labels, predictions)
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            
            return {
                'f1': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train model
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        training_time = time.time() - start_time
        
        # Extract metrics and add training time
        metrics = {
            'f1': eval_results.get('eval_f1', 0.0),
            'accuracy': eval_results.get('eval_accuracy', 0.0),
            'precision': eval_results.get('eval_precision', 0.0),
            'recall': eval_results.get('eval_recall', 0.0),
            'training_time': training_time
        }
        
        # Cleanup
        del model, trainer, train_dataset, val_dataset
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return metrics
    
    def optimize_roberta(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Perform grid search optimization for RoBERTa.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Optimization results dictionary
        """
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available. Skipping RoBERTa optimization.")
            return {'error': 'Transformers not available'}
        
        print("\n🤖 ROBERTA GRID SEARCH")
        print("=" * 50)
        
        start_time = time.time()
        
        # Get parameter grid
        param_grid = self.get_roberta_param_grid()
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))
        
        print(f"📊 Parameter grid combinations: {len(all_combinations)}")
        print(f"📊 With 3-fold CV: {len(all_combinations) * 3} model fits")
        print(f"📊 Estimated time: {len(all_combinations) * 3 * 4:.1f} minutes")
        
        # Set up stratified cross-validation
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        # Store results for each combination
        cv_results = []
        best_score = 0.0
        best_params = None
        
        print("\n🚀 Starting RoBERTa grid search...")
        
        for i, combination in enumerate(all_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            print(f"\n🔄 Configuration {i+1}/{len(all_combinations)}: {params}")
            
            # Perform cross-validation for this configuration
            cv_scores = []
            cv_times = []
            
            for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
                print(f"  📊 Fold {fold+1}/3...")
                
                # Split data for this fold
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Train and evaluate
                try:
                    fold_metrics = self.train_single_roberta_config(
                        params, X_fold_train, y_fold_train, X_fold_val, y_fold_val
                    )
                    cv_scores.append(fold_metrics['f1'])
                    cv_times.append(fold_metrics['training_time'])
                    print(f"    F1: {fold_metrics['f1']:.4f}, Time: {fold_metrics['training_time']:.1f}s")
                except Exception as e:
                    print(f"    ❌ Fold failed: {str(e)}")
                    cv_scores.append(0.0)
                    cv_times.append(0.0)
            
            # Calculate mean CV score
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            mean_time = np.mean(cv_times)
            
            # Store results
            cv_results.append({
                'params': params,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'cv_scores': cv_scores,
                'mean_training_time': mean_time
            })
            
            print(f"  📈 CV F1: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
            
            # Update best parameters
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_params = params.copy()
                print(f"  🏆 New best score: {best_score:.4f}")
        
        # Train final model with best parameters on full training set
        print(f"\n🎯 Training final model with best parameters...")
        print(f"🏆 Best params: {best_params}")
        
        try:
            # Train on full training set for final evaluation
            final_metrics = self.train_single_roberta_config(
                best_params, X_train, y_train, X_test, y_test
            )
            
            optimization_time = time.time() - start_time
            
            results = {
                'model_type': 'RoBERTa',
                'best_params': best_params,
                'best_cv_score': best_score,
                'test_metrics': final_metrics,
                'optimization_time': optimization_time,
                'total_combinations': len(all_combinations),
                'cv_results': cv_results
            }
            
            print(f"\n✅ RoBERTa grid search completed in {optimization_time:.2f} seconds")
            print(f"🏆 Best CV F1 Score: {best_score:.4f}")
            print(f"🎯 Test F1 Score: {final_metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"❌ Final model training failed: {str(e)}")
            results = {
                'model_type': 'RoBERTa',
                'best_params': best_params,
                'best_cv_score': best_score,
                'error': str(e),
                'cv_results': cv_results
            }
        
        return results
    
    def compare_results(self, logreg_results: Dict[str, Any], 
                       roberta_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare optimization results between models.
        
        Args:
            logreg_results: LogisticRegression results
            roberta_results: RoBERTa results
            
        Returns:
            Comparison analysis
        """
        print("\n📊 OPTIMIZATION RESULTS COMPARISON")
        print("=" * 50)
        
        comparison = {
            'models_compared': ['LogisticRegression', 'RoBERTa'],
            'optimization_times': {},
            'best_scores': {},
            'test_performances': {},
            'parameter_space_sizes': {},
            'winner': None
        }
        
        # Extract key metrics
        for model_name, results in [('LogisticRegression', logreg_results), ('RoBERTa', roberta_results)]:
            if 'error' not in results:
                comparison['optimization_times'][model_name] = results.get('optimization_time', 0)
                comparison['best_scores'][model_name] = results.get('best_cv_score', 0)
                comparison['test_performances'][model_name] = results.get('test_metrics', {})
                comparison['parameter_space_sizes'][model_name] = results.get('total_combinations', 0)
        
        # Determine winner based on test F1 score
        if ('LogisticRegression' in comparison['test_performances'] and 
            'RoBERTa' in comparison['test_performances']):
            
            lr_f1 = comparison['test_performances']['LogisticRegression'].get('f1', 0)
            roberta_f1 = comparison['test_performances']['RoBERTa'].get('f1', 0)
            
            if roberta_f1 > lr_f1:
                comparison['winner'] = 'RoBERTa'
                comparison['performance_gap'] = roberta_f1 - lr_f1
            else:
                comparison['winner'] = 'LogisticRegression'
                comparison['performance_gap'] = lr_f1 - roberta_f1
            
            print(f"🏆 Winner: {comparison['winner']}")
            print(f"📈 Performance gap: {comparison['performance_gap']:.4f} F1 points")
        
        # Print comparison table
        print(f"\n📋 RESULTS SUMMARY:")
        print(f"{'Model':<20} {'CV F1':<10} {'Test F1':<10} {'Time (min)':<12} {'Combinations':<12}")
        print("-" * 70)
        
        for model_name in ['LogisticRegression', 'RoBERTa']:
            if model_name in comparison['test_performances']:
                cv_f1 = comparison['best_scores'][model_name]
                test_f1 = comparison['test_performances'][model_name].get('f1', 0)
                opt_time = comparison['optimization_times'][model_name] / 60
                combinations = comparison['parameter_space_sizes'][model_name]
                
                print(f"{model_name:<20} {cv_f1:<10.4f} {test_f1:<10.4f} {opt_time:<12.1f} {combinations:<12}")
        
        return comparison
    
    def save_results(self, logreg_results: Dict[str, Any], roberta_results: Dict[str, Any],
                    comparison: Dict[str, Any]):
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results for JSON serialization
        all_results = {
            'timestamp': timestamp,
            'logistic_regression': self._serialize_results(logreg_results),
            'roberta': self._serialize_results(roberta_results),
            'comparison': comparison,
            'metadata': {
                'optimization_method': 'GridSearchCV',
                'cv_folds': 3,
                'random_state': self.random_state,
                'device_used': str(self.device) if TRANSFORMERS_AVAILABLE else 'cpu'
            }
        }
        
        # Save main results
        results_file = self.config.output_dir / f"gridsearch_optimization_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"💾 Results saved: {results_file}")
        
        # Generate summary report
        self._generate_summary_report(all_results, timestamp)
    
    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-serializable objects from results."""
        serialized = results.copy()
        
        # Remove non-serializable objects
        serialized.pop('best_model', None)
        serialized.pop('predictions', None)
        serialized.pop('probabilities', None)
        
        # Convert numpy arrays in cv_results if present
        if 'cv_results' in serialized and isinstance(serialized['cv_results'], dict):
            for key, value in serialized['cv_results'].items():
                if hasattr(value, 'tolist'):
                    serialized['cv_results'][key] = value.tolist()
        
        return serialized
    
    def _generate_summary_report(self, results: Dict[str, Any], timestamp: str):
        """Generate a markdown summary report."""
        report_file = self.config.output_dir / f"gridsearch_optimization_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# SpoilerShield: Grid Search Hyperparameter Optimization Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview
            f.write("## 🔍 Optimization Overview\n\n")
            f.write("This report summarizes the results of systematic grid search hyperparameter optimization\n")
            f.write("for both traditional ML (LogisticRegression + TF-IDF) and transformer (RoBERTa) models.\n\n")
            
            # Method
            f.write("## 🛠️ Methodology\n\n")
            f.write("- **Optimization Method**: GridSearchCV with 3-fold Stratified Cross-Validation\n")
            f.write("- **Evaluation Metric**: F1 Score (primary), with additional metrics tracked\n")
            f.write("- **Data Split**: 80% train, 20% test with stratification\n")
            f.write("- **Parallel Processing**: Multi-core CPU for LogReg, GPU for RoBERTa\n\n")
            
            # Results summary
            if 'comparison' in results:
                comparison = results['comparison']
                f.write("## 🏆 Results Summary\n\n")
                
                if comparison.get('winner'):
                    f.write(f"**Overall Winner:** {comparison['winner']}\n")
                    f.write(f"**Performance Gap:** {comparison.get('performance_gap', 0):.4f} F1 points\n\n")
                
                # Results table
                f.write("| Model | CV F1 Score | Test F1 Score | Optimization Time | Parameter Combinations |\n")
                f.write("|-------|-------------|---------------|-------------------|------------------------|\n")
                
                for model in ['LogisticRegression', 'RoBERTa']:
                    if model in comparison.get('test_performances', {}):
                        cv_f1 = comparison['best_scores'].get(model, 0)
                        test_f1 = comparison['test_performances'][model].get('f1', 0)
                        opt_time = comparison['optimization_times'].get(model, 0) / 60
                        combinations = comparison['parameter_space_sizes'].get(model, 0)
                        
                        f.write(f"| {model} | {cv_f1:.4f} | {test_f1:.4f} | {opt_time:.1f} min | {combinations} |\n")
                f.write("\n")
            
            # Best parameters
            f.write("## 🎯 Optimal Hyperparameters\n\n")
            
            for model_name, model_key in [('LogisticRegression', 'logistic_regression'), ('RoBERTa', 'roberta')]:
                if model_key in results and 'best_params' in results[model_key]:
                    f.write(f"### {model_name}\n\n")
                    best_params = results[model_key]['best_params']
                    for param, value in best_params.items():
                        f.write(f"- **{param}**: {value}\n")
                    f.write("\n")
            
            # Next steps
            f.write("## 🚀 Next Steps\n\n")
            f.write("1. **Genetic Algorithm Comparison**: Implement GA optimization with same parameter space\n")
            f.write("2. **A/B Analysis**: Compare GridSearch vs GA efficiency and effectiveness\n")
            f.write("3. **Production Integration**: Deploy best-performing configuration\n")
            f.write("4. **Ensemble Methods**: Combine optimized models for potential performance gains\n")
        
        print(f"📋 Summary report saved: {report_file}")
    
    def run_comprehensive_optimization(self, df_reviews: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive grid search optimization for both models.
        
        Args:
            df_reviews: DataFrame with review data
            
        Returns:
            Complete optimization results
        """
        print("🚀 STARTING COMPREHENSIVE GRID SEARCH OPTIMIZATION")
        print("=" * 70)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df_reviews)
        
        # Optimize LogisticRegression
        print("\n" + "="*70)
        logreg_results = self.optimize_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Optimize RoBERTa
        print("\n" + "="*70)
        roberta_results = self.optimize_roberta(X_train, y_train, X_test, y_test)
        
        # Compare results
        print("\n" + "="*70)
        comparison = self.compare_results(logreg_results, roberta_results)
        
        # Save results
        self.save_results(logreg_results, roberta_results, comparison)
        
        print("\n🎉 GRID SEARCH OPTIMIZATION COMPLETE!")
        print("=" * 70)
        
        return {
            'logistic_regression': logreg_results,
            'roberta': roberta_results,
            'comparison': comparison
        }


def main():
    """Main execution function."""
    print("🔍 SPOILERSHIELD: GRID SEARCH HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Initialize configuration
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
    
    # Initialize optimizer
    optimizer = GridSearchOptimizer(config)
    
    # Run comprehensive optimization
    results = optimizer.run_comprehensive_optimization(df_reviews)
    
    print(f"\n🎯 OPTIMIZATION SUMMARY:")
    print(f"📁 Results saved to: {config.output_dir}")
    
    # Display winner
    if results['comparison'].get('winner'):
        winner = results['comparison']['winner']
        gap = results['comparison'].get('performance_gap', 0)
        print(f"🏆 Best Model: {winner}")
        print(f"📈 Performance Advantage: {gap:.4f} F1 points")
    
    print(f"\n✅ Ready for Genetic Algorithm A/B comparison!")


if __name__ == "__main__":
    main()
