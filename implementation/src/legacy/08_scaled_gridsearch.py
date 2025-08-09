#!/usr/bin/env python3
"""
SpoilerShield: Scaled GridSearch for A/B Testing
================================================

Moderate-scale grid search with multi-objective optimization:
- LogisticRegression: 288 combinations (4×6×3×4) 
- RoBERTa: 48 combinations (2×6×2×2)
- Multi-objective: F1 + AUC + efficiency
- Estimated runtime: ~20-30 minutes

Author: SpoilerShield Development Team
Date: 2025-01-07
"""

from __future__ import annotations
import time
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, make_scorer

# Transformers (conditional import)
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
        set_seed as transformers_set_seed
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Only LogisticRegression grid search will run.")

# Project-specific
from utils.env_config import EnvConfig
from eda.data_loader import DataLoader


def set_all_seeds(seed: int):
    """Set seeds for all random number generators for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if TRANSFORMERS_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        transformers_set_seed(seed)
    
    print(f"🔒 All random seeds set to: {seed}")


@dataclass
class ScaledGridResult:
    model_type: str
    best_params: Dict[str, Any]
    best_cv_f1: float
    best_cv_auc: float
    test_metrics: Dict[str, float]
    multi_objective_score: float
    optimization_time: float
    param_combos: int
    n_splits: int
    total_evals: int
    cv_indices: List[Tuple[np.ndarray, np.ndarray]]
    best_so_far_f1: List[float]
    best_so_far_auc: List[float]


class ScaledGridSearchOptimizer:
    """
    Scaled-up GridSearch for showcasing GA advantages.
    
    Features:
    - Moderate parameter space (288 LogReg, 48 RoBERTa combinations)
    - Multi-objective optimization (F1 + AUC + efficiency)
    - Parallel execution with n_jobs=-1
    - Full reproducibility
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        set_all_seeds(random_state)
        
        # Multi-objective weights
        self.f1_weight = 0.4
        self.auc_weight = 0.5      # Stronger weight on AUC
        self.efficiency_weight = 0.1  # Weaker weight on efficiency
        
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")
        
        print("🚀 SPOILERSHIELD: SCALED GRID SEARCH")
        print("=" * 60)
    
    def prepare_data(self, df_reviews: pd.DataFrame):
        """Prepare data with consistent splits."""
        texts = df_reviews['review_text'].values
        labels = df_reviews['is_spoiler'].values
        return train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=self.random_state)
    
    def logistic_param_grid(self) -> Dict[str, List]:
        """
        Scaled LogisticRegression parameter grid.
        
        Returns:
            288 combinations (4×6×3×4)
        """
        return {
            'tfidf__max_features': [5000, 10000, 20000, 40000],        # 4 values
            'classifier__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],         # 6 values  
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],         # 3 values
            'tfidf__ngram_range': [(1,1), (1,2), (1,3), (2,3)],       # 4 values
            
            # Fixed parameters for consistency
            'tfidf__min_df': [2],
            'tfidf__max_df': [0.95],
            'tfidf__sublinear_tf': [True],
            'classifier__class_weight': ['balanced'],
            'classifier__max_iter': [3000],
            'classifier__l1_ratio': [0.5],  # For elasticnet
        }
    
    def roberta_param_grid(self) -> Dict[str, List]:
        """
        Scaled RoBERTa parameter grid.
        
        Returns:
            48 combinations (2×6×2×2)
        """
        return {
            'model_name': ['roberta-base', 'roberta-large'],           # 2 values
            'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4],   # 6 values
            'num_train_epochs': [3, 4],                               # 2 values
            'max_length': [256, 512],                                 # 2 values
            
            # Fixed parameters for consistency
            'per_device_train_batch_size': [8],  # Reduced for large model memory
            'weight_decay': [0.01],
            'warmup_ratio': [0.1],
            'dropout_rate': [0.1],
        }
    
    def calculate_multi_objective_score(self, f1: float, auc: float, time_seconds: float) -> float:
        """
        Calculate multi-objective fitness score.
        
        Args:
            f1: F1 score (0-1)
            auc: ROC AUC score (0-1)  
            time_seconds: Training time in seconds
            
        Returns:
            Multi-objective score (higher is better)
        """
        # Normalize time (log scale to handle wide range)
        normalized_time = 1.0 / (1.0 + np.log(1.0 + time_seconds / 60.0))  # Convert to minutes
        
        # Weighted combination
        score = (self.f1_weight * f1 + 
                self.auc_weight * auc + 
                self.efficiency_weight * normalized_time)
        
        return score
    
    def optimize_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> ScaledGridResult:
        """Scaled LogisticRegression optimization."""
        print(f"\n🔍 SCALED LOGISTIC REGRESSION GRID SEARCH")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create pipeline with saga solver for all penalties
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode')),
            ('classifier', LogisticRegression(random_state=self.random_state, solver='saga', n_jobs=1))
        ])
        
        param_grid = self.logistic_param_grid()
        param_combos = 4 * 6 * 3 * 4  # 288 combinations
        n_splits = 3
        total_evals = param_combos * n_splits
        
        print(f"📊 Parameter combinations: {param_combos}")
        print(f"📊 Total evaluations: {total_evals}")
        print(f"📊 Estimated time: ~20-25 minutes")
        
        # Fixed CV folds for sharing with GA
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_indices = list(skf.split(X_train, y_train))
        
        # Multi-objective scoring
        scorers = {
            'f1': make_scorer(f1_score),
            'roc_auc': 'roc_auc'
        }
        
        # Run grid search
        print(f"\n🚀 Starting scaled grid search...")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_indices,
            scoring=scorers,
            refit='f1',  # Primary metric for refitting
            n_jobs=-1,   # Parallel execution
            verbose=1,
            return_train_score=False
        )
        
        grid_search.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        
        # Get best model and evaluate
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive test metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
        }
        
        # Extract CV results for best-so-far tracking
        cv_results = grid_search.cv_results_
        mean_f1_scores = cv_results['mean_test_f1']
        mean_auc_scores = cv_results['mean_test_roc_auc']
        
        # Build best-so-far tracking (assuming GridSearchCV order)
        best_so_far_f1 = []
        best_so_far_auc = []
        cur_best_f1 = -1.0
        cur_best_auc = -1.0
        
        for i in range(len(mean_f1_scores)):
            cur_best_f1 = max(cur_best_f1, float(mean_f1_scores[i]))
            cur_best_auc = max(cur_best_auc, float(mean_auc_scores[i]))
            # Expand by n_splits for eval-count granularity
            for _ in range(n_splits):
                best_so_far_f1.append(cur_best_f1)
                best_so_far_auc.append(cur_best_auc)
        
        # Calculate multi-objective score
        best_cv_f1 = grid_search.best_score_  # This is F1 since refit='f1'
        best_cv_auc = max(mean_auc_scores)  # Best AUC from CV
        multi_objective_score = self.calculate_multi_objective_score(
            test_metrics['f1'], test_metrics['roc_auc'], optimization_time
        )
        
        print(f"\n✅ Grid search completed in {optimization_time:.1f} seconds")
        print(f"🏆 Best CV F1: {best_cv_f1:.4f}")
        print(f"🏆 Best CV AUC: {best_cv_auc:.4f}")
        print(f"🎯 Test F1: {test_metrics['f1']:.4f}")
        print(f"🎯 Test AUC: {test_metrics['roc_auc']:.4f}")
        print(f"⚖️ Multi-objective score: {multi_objective_score:.4f}")
        print(f"📊 Best parameters: {grid_search.best_params_}")
        
        return ScaledGridResult(
            model_type='LogisticRegression_Scaled',
            best_params=grid_search.best_params_,
            best_cv_f1=best_cv_f1,
            best_cv_auc=best_cv_auc,
            test_metrics=test_metrics,
            multi_objective_score=multi_objective_score,
            optimization_time=optimization_time,
            param_combos=param_combos,
            n_splits=n_splits,
            total_evals=total_evals,
            cv_indices=cv_indices,
            best_so_far_f1=best_so_far_f1,
            best_so_far_auc=best_so_far_auc
        )
    
    def optimize_roberta(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        cv_indices: List[Tuple[np.ndarray, np.ndarray]]) -> ScaledGridResult:
        """Scaled RoBERTa optimization."""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available. Skipping RoBERTa optimization.")
            return None
        
        print(f"\n🤖 SCALED ROBERTA GRID SEARCH")
        print("=" * 60)
        
        start_time = time.time()
        param_grid = self.roberta_param_grid()
        
        # Calculate combinations
        param_combos = 2 * 6 * 2 * 2  # 48 combinations
        n_splits = len(cv_indices)
        total_evals = param_combos * n_splits
        
        print(f"📊 Parameter combinations: {param_combos}")
        print(f"📊 Total evaluations: {total_evals}")
        print(f"📊 Estimated time: ~45-60 minutes")
        
        # Generate all parameter combinations
        model_names = param_grid['model_name']
        learning_rates = param_grid['learning_rate'] 
        epochs = param_grid['num_train_epochs']
        max_lengths = param_grid['max_length']
        
        all_results = []
        best_score = -1.0
        best_params = None
        best_f1 = -1.0
        best_auc = -1.0
        
        # Track best-so-far
        best_so_far_f1 = []
        best_so_far_auc = []
        eval_count = 0
        
        print(f"\n🚀 Starting RoBERTa parameter search...")
        
        for model_name in model_names:
            for lr in learning_rates:
                for epoch in epochs:
                    for max_len in max_lengths:
                        params = {
                            'model_name': model_name,
                            'learning_rate': lr,
                            'num_train_epochs': epoch,
                            'max_length': max_len,
                            'per_device_train_batch_size': param_grid['per_device_train_batch_size'][0],
                            'weight_decay': param_grid['weight_decay'][0],
                            'warmup_ratio': param_grid['warmup_ratio'][0],
                            'dropout_rate': param_grid['dropout_rate'][0]
                        }
                        
                        print(f"\n🔄 Testing: {model_name}, LR={lr}, Epochs={epoch}, MaxLen={max_len}")
                        
                        # Cross-validation
                        cv_f1_scores = []
                        cv_auc_scores = []
                        fold_times = []
                        
                        for fold_idx, (train_idx, val_idx) in enumerate(cv_indices):
                            print(f"  📊 Fold {fold_idx+1}/{n_splits}...")
                            
                            fold_start = time.time()
                            
                            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                            
                            try:
                                fold_f1, fold_auc = self._train_roberta_fold(
                                    params, X_fold_train, y_fold_train, X_fold_val, y_fold_val
                                )
                                cv_f1_scores.append(fold_f1)
                                cv_auc_scores.append(fold_auc)
                                fold_times.append(time.time() - fold_start)
                                
                                print(f"    F1: {fold_f1:.4f}, AUC: {fold_auc:.4f}")
                                
                                # Update best-so-far tracking
                                eval_count += 1
                                current_best_f1 = max(best_f1, fold_f1)
                                current_best_auc = max(best_auc, fold_auc)
                                best_so_far_f1.append(current_best_f1)
                                best_so_far_auc.append(current_best_auc)
                                
                            except Exception as e:
                                print(f"    ❌ Fold failed: {str(e)}")
                                cv_f1_scores.append(0.0)
                                cv_auc_scores.append(0.5)
                                fold_times.append(60.0)  # Penalty time
                                
                                eval_count += 1
                                best_so_far_f1.append(best_f1)
                                best_so_far_auc.append(best_auc)
                        
                        # Calculate mean scores
                        mean_f1 = np.mean(cv_f1_scores)
                        mean_auc = np.mean(cv_auc_scores)
                        mean_time = np.mean(fold_times)
                        
                        # Multi-objective score
                        multi_obj_score = self.calculate_multi_objective_score(mean_f1, mean_auc, mean_time)
                        
                        all_results.append({
                            'params': params,
                            'cv_f1': mean_f1,
                            'cv_auc': mean_auc,
                            'cv_time': mean_time,
                            'multi_objective': multi_obj_score
                        })
                        
                        print(f"  📈 CV F1: {mean_f1:.4f}, CV AUC: {mean_auc:.4f}, Score: {multi_obj_score:.4f}")
                        
                        # Track overall best
                        if multi_obj_score > best_score:
                            best_score = multi_obj_score
                            best_params = params.copy()
                            best_f1 = mean_f1
                            best_auc = mean_auc
                            print(f"  🏆 New best multi-objective score: {best_score:.4f}")
        
        # Train final model with best parameters
        print(f"\n🎯 Training final model with best params...")
        try:
            final_f1, final_auc = self._train_roberta_fold(best_params, X_train, y_train, X_test, y_test)
            
            test_metrics = {
                'f1': final_f1,
                'roc_auc': final_auc,
                'accuracy': 0.0,  # Not calculated for speed
                'precision': 0.0,
                'recall': 0.0,
                'specificity': 0.0
            }
            
            optimization_time = time.time() - start_time
            
            print(f"\n✅ RoBERTa optimization completed in {optimization_time:.1f} seconds")
            print(f"🏆 Best CV F1: {best_f1:.4f}")
            print(f"🏆 Best CV AUC: {best_auc:.4f}")
            print(f"🎯 Test F1: {final_f1:.4f}")
            print(f"🎯 Test AUC: {final_auc:.4f}")
            print(f"⚖️ Multi-objective score: {best_score:.4f}")
            
            return ScaledGridResult(
                model_type='RoBERTa_Scaled',
                best_params=best_params,
                best_cv_f1=best_f1,
                best_cv_auc=best_auc,
                test_metrics=test_metrics,
                multi_objective_score=best_score,
                optimization_time=optimization_time,
                param_combos=param_combos,
                n_splits=n_splits,
                total_evals=total_evals,
                cv_indices=cv_indices,
                best_so_far_f1=best_so_far_f1,
                best_so_far_auc=best_so_far_auc
            )
            
        except Exception as e:
            print(f"❌ Final training failed: {str(e)}")
            return None
    
    def _train_roberta_fold(self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        """Train a single RoBERTa fold and return F1 and AUC scores."""
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
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=Path.cwd() / 'temp_roberta_scaled',
            num_train_epochs=params['num_train_epochs'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=params['per_device_train_batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            warmup_ratio=params['warmup_ratio'],
            logging_steps=200,
            eval_strategy="epoch",
            save_strategy="no",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            seed=self.random_state,
            data_seed=self.random_state,
            dataloader_num_workers=0,
        )
        
        # Compute metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            # Calculate probabilities for AUC
            probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=1)[:, 1].numpy()
            
            f1 = f1_score(labels, predictions)
            auc = roc_auc_score(labels, probs)
            
            return {'f1': f1, 'roc_auc': auc}
        
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
        
        return eval_results.get('eval_f1', 0.0), eval_results.get('eval_roc_auc', 0.5)
    
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


def main():
    """Main execution function."""
    print("🚀 SPOILERSHIELD: SCALED GRID SEARCH")
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
    optimizer = ScaledGridSearchOptimizer()
    X_train, X_test, y_train, y_test = optimizer.prepare_data(df_reviews)
    
    # LogisticRegression optimization
    logreg_results = optimizer.optimize_logistic_regression(X_train, y_train, X_test, y_test)
    
    # RoBERTa optimization (using same CV folds)
    roberta_results = optimizer.optimize_roberta(X_train, y_train, X_test, y_test, logreg_results.cv_indices)
    
    # Save results
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'method': 'Scaled_GridSearch',
        'logistic_regression': {
            'model_type': logreg_results.model_type,
            'best_params': logreg_results.best_params,
            'best_cv_f1': logreg_results.best_cv_f1,
            'best_cv_auc': logreg_results.best_cv_auc,
            'test_metrics': logreg_results.test_metrics,
            'multi_objective_score': logreg_results.multi_objective_score,
            'optimization_time': logreg_results.optimization_time,
            'total_evals': logreg_results.total_evals,
        },
        'roberta': {
            'model_type': roberta_results.model_type if roberta_results else 'Not_Available',
            'best_params': roberta_results.best_params if roberta_results else {},
            'best_cv_f1': roberta_results.best_cv_f1 if roberta_results else 0.0,
            'best_cv_auc': roberta_results.best_cv_auc if roberta_results else 0.0,
            'test_metrics': roberta_results.test_metrics if roberta_results else {},
            'multi_objective_score': roberta_results.multi_objective_score if roberta_results else 0.0,
            'optimization_time': roberta_results.optimization_time if roberta_results else 0.0,
            'total_evals': roberta_results.total_evals if roberta_results else 0,
        } if roberta_results else {'error': 'Transformers not available'}
    }
    
    results_file = config.output_dir / f"scaled_gridsearch_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_file}")
    
    print(f"\n📊 SCALED GRID SEARCH SUMMARY:")
    print(f"  LogReg Multi-obj: {logreg_results.multi_objective_score:.4f}")
    if roberta_results:
        print(f"  RoBERTa Multi-obj: {roberta_results.multi_objective_score:.4f}")


if __name__ == "__main__":
    main()
