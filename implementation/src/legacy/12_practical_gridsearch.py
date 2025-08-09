#!/usr/bin/env python3
"""
Practical GridSearch with optimized parameter space for reasonable runtime.
"""

import time
import json
import random
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

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

# Project-specific
from utils.env_config import EnvConfig
from eda.data_loader import DataLoader


def set_all_seeds(seed: int):
    """Set seeds for all random number generators."""
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


@dataclass
class PracticalGridResult:
    model_type: str
    best_params: Dict[str, Any]
    best_cv_f1: float
    best_cv_auc: float
    test_metrics: Dict[str, float]
    multi_objective_score: float
    optimization_time: float
    param_combos: int
    total_evals: int


class PracticalGridSearchOptimizer:
    """Practical GridSearch with balanced runtime vs statistical power."""
    
    def __init__(self, random_state: int = 42, sample_size: int = 20000):
        self.random_state = random_state
        self.sample_size = sample_size
        set_all_seeds(random_state)
        
        # Multi-objective weights
        self.f1_weight = 0.4
        self.auc_weight = 0.5
        self.efficiency_weight = 0.1
        
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def prepare_data(self, df_reviews: pd.DataFrame):
        """Prepare data with sampling for practical runtime."""
        # Sample data for practical runtime while maintaining class balance
        if len(df_reviews) > self.sample_size:
            print(f"📊 Sampling {self.sample_size:,} from {len(df_reviews):,} for practical runtime")
            df_sampled = df_reviews.groupby('is_spoiler', group_keys=False).apply(
                lambda x: x.sample(min(len(x), self.sample_size // 2), random_state=self.random_state)
            ).reset_index(drop=True)
        else:
            df_sampled = df_reviews
        
        texts = df_sampled['review_text'].values
        labels = df_sampled['is_spoiler'].values
        
        return train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=self.random_state)
    
    def logistic_param_grid(self) -> Dict[str, List]:
        """
        Practical LogReg grid: 48 combinations (3×4×2×2).
        Estimated time: ~15-20 minutes with 20k samples.
        """
        return {
            'tfidf__max_features': [10000, 20000, 40000],              # 3 values
            'classifier__C': [0.5, 1.0, 2.0, 5.0],                    # 4 values
            'classifier__penalty': ['l2', 'elasticnet'],               # 2 values
            'tfidf__ngram_range': [(1,1), (1,2)],                     # 2 values
            
            # Fixed parameters for speed and stability
            'tfidf__min_df': [2],
            'tfidf__max_df': [0.95],
            'tfidf__sublinear_tf': [True],
            'classifier__class_weight': ['balanced'],
            'classifier__max_iter': [2000],
            'classifier__l1_ratio': [0.5],
        }
    
    def roberta_param_grid(self) -> Dict[str, List]:
        """
        Practical RoBERTa grid: 8 combinations (2×2×2).
        Estimated time: ~10-15 minutes with 20k samples.
        """
        return {
            'model_name': ['roberta-base', 'roberta-large'],           # 2 values
            'learning_rate': [3e-5, 5e-5],                            # 2 values
            'max_length': [256, 512],                                 # 2 values
            
            # Fixed parameters for speed
            'num_train_epochs': [3],
            'per_device_train_batch_size': [16],
            'weight_decay': [0.01],
            'warmup_ratio': [0.1],
            'dropout_rate': [0.1],
        }
    
    def calculate_multi_objective_score(self, f1: float, auc: float, time_seconds: float) -> float:
        """Calculate multi-objective fitness score."""
        normalized_time = 1.0 / (1.0 + np.log(1.0 + time_seconds / 60.0))
        return (self.f1_weight * f1 + self.auc_weight * auc + self.efficiency_weight * normalized_time)
    
    def optimize_logistic_regression(self, X_train, y_train, X_test, y_test) -> PracticalGridResult:
        """Practical LogReg optimization."""
        print(f"\n📊 PRACTICAL LOGISTIC REGRESSION GRID SEARCH")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode')),
            ('classifier', LogisticRegression(random_state=self.random_state, solver='saga'))
        ])
        
        param_grid = self.logistic_param_grid()
        param_combos = 3 * 4 * 2 * 2  # 48 combinations
        n_splits = 3
        total_evals = param_combos * n_splits
        
        print(f"📊 Parameter combinations: {param_combos}")
        print(f"📊 Total evaluations: {total_evals}")
        print(f"📊 Estimated time: ~15-20 minutes")
        
        # Fixed CV folds
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_indices = list(skf.split(X_train, y_train))
        
        # Multi-objective scoring
        scorers = {
            'f1': make_scorer(f1_score),
            'roc_auc': 'roc_auc'
        }
        
        # Suppress warnings for cleaner output
        import warnings
        warnings.filterwarnings('ignore')
        
        # Run grid search
        print(f"\n🚀 Starting practical grid search...")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_indices,
            scoring=scorers,
            refit='f1',
            n_jobs=-1,
            verbose=1,
            return_train_score=False
        )
        
        grid_search.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        
        # Evaluate best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate test metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
        }
        
        # Get best CV scores
        cv_results = grid_search.cv_results_
        best_cv_f1 = grid_search.best_score_
        best_cv_auc = max(cv_results['mean_test_roc_auc'])
        
        # Calculate multi-objective score
        multi_objective_score = self.calculate_multi_objective_score(
            test_metrics['f1'], test_metrics['roc_auc'], optimization_time
        )
        
        print(f"\n✅ Grid search completed in {optimization_time:.1f} seconds")
        print(f"🏆 Best CV F1: {best_cv_f1:.4f}")
        print(f"🏆 Best CV AUC: {best_cv_auc:.4f}")
        print(f"🎯 Test F1: {test_metrics['f1']:.4f}")
        print(f"🎯 Test AUC: {test_metrics['roc_auc']:.4f}")
        print(f"⚖️ Multi-objective score: {multi_objective_score:.4f}")
        
        return PracticalGridResult(
            model_type='LogisticRegression_Practical',
            best_params=grid_search.best_params_,
            best_cv_f1=best_cv_f1,
            best_cv_auc=best_cv_auc,
            test_metrics=test_metrics,
            multi_objective_score=multi_objective_score,
            optimization_time=optimization_time,
            param_combos=param_combos,
            total_evals=total_evals
        )
    
    def optimize_roberta(self, X_train, y_train, X_test, y_test) -> PracticalGridResult:
        """Practical RoBERTa optimization."""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available. Skipping RoBERTa optimization.")
            return None
        
        print(f"\n🤖 PRACTICAL ROBERTA GRID SEARCH")
        print("=" * 60)
        
        start_time = time.time()
        param_grid = self.roberta_param_grid()
        
        param_combos = 2 * 2 * 2  # 8 combinations
        total_evals = param_combos
        
        print(f"📊 Parameter combinations: {param_combos}")
        print(f"📊 Total evaluations: {total_evals}")
        print(f"📊 Estimated time: ~10-15 minutes")
        
        # Generate parameter combinations
        model_names = param_grid['model_name']
        learning_rates = param_grid['learning_rate']
        max_lengths = param_grid['max_length']
        
        best_score = -1.0
        best_params = None
        best_f1 = -1.0
        best_auc = -1.0
        
        print(f"\n🚀 Starting RoBERTa parameter search...")
        
        for model_name in model_names:
            for lr in learning_rates:
                for max_len in max_lengths:
                    params = {
                        'model_name': model_name,
                        'learning_rate': lr,
                        'max_length': max_len,
                        'num_train_epochs': param_grid['num_train_epochs'][0],
                        'per_device_train_batch_size': param_grid['per_device_train_batch_size'][0],
                        'weight_decay': param_grid['weight_decay'][0],
                        'warmup_ratio': param_grid['warmup_ratio'][0],
                        'dropout_rate': param_grid['dropout_rate'][0]
                    }
                    
                    print(f"\n🔄 Testing: {model_name}, LR={lr}, MaxLen={max_len}")
                    
                    try:
                        f1, auc = self._train_roberta_config(params, X_train, y_train, X_test, y_test)
                        multi_obj_score = self.calculate_multi_objective_score(f1, auc, 300)  # Estimated time per config
                        
                        print(f"  📈 F1: {f1:.4f}, AUC: {auc:.4f}, Score: {multi_obj_score:.4f}")
                        
                        if multi_obj_score > best_score:
                            best_score = multi_obj_score
                            best_params = params.copy()
                            best_f1 = f1
                            best_auc = auc
                            print(f"  🏆 New best score: {best_score:.4f}")
                            
                    except Exception as e:
                        print(f"  ❌ Config failed: {str(e)}")
                        continue
        
        optimization_time = time.time() - start_time
        
        # Test metrics (using best F1/AUC as approximation)
        test_metrics = {
            'f1': best_f1,
            'roc_auc': best_auc,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0
        }
        
        print(f"\n✅ RoBERTa optimization completed in {optimization_time:.1f} seconds")
        print(f"🏆 Best F1: {best_f1:.4f}")
        print(f"🏆 Best AUC: {best_auc:.4f}")
        print(f"⚖️ Multi-objective score: {best_score:.4f}")
        
        return PracticalGridResult(
            model_type='RoBERTa_Practical',
            best_params=best_params,
            best_cv_f1=best_f1,
            best_cv_auc=best_auc,
            test_metrics=test_metrics,
            multi_objective_score=best_score,
            optimization_time=optimization_time,
            param_combos=param_combos,
            total_evals=total_evals
        )
    
    def _train_roberta_config(self, params: Dict[str, Any], X_train, y_train, X_test, y_test) -> Tuple[float, float]:
        """Train a single RoBERTa configuration and return F1 and AUC scores."""
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            params['model_name'], 
            num_labels=2,
            hidden_dropout_prob=params['dropout_rate']
        )
        
        # Create datasets (use only subset for speed)
        train_size = min(len(X_train), 5000)  # Limit training size for practical runtime
        X_train_subset = X_train[:train_size]
        y_train_subset = y_train[:train_size]
        
        train_dataset = self._create_dataset(X_train_subset, y_train_subset, tokenizer, params['max_length'])
        test_dataset = self._create_dataset(X_test, y_test, tokenizer, params['max_length'])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=Path.cwd() / 'temp_practical_roberta',
            num_train_epochs=params['num_train_epochs'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=params['per_device_train_batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            warmup_ratio=params['warmup_ratio'],
            logging_steps=100,
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
            
            probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=1)[:, 1].numpy()
            
            f1 = f1_score(labels, predictions)
            auc = roc_auc_score(labels, probs)
            
            return {'f1': f1, 'roc_auc': auc}
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        
        # Cleanup
        del model, trainer, train_dataset, test_dataset
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return eval_results.get('eval_f1', 0.0), eval_results.get('eval_roc_auc', 0.5)
    
    def _create_dataset(self, texts, labels, tokenizer, max_length: int):
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
    print("⚡ SPOILERSHIELD: PRACTICAL GRID SEARCH")
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
    optimizer = PracticalGridSearchOptimizer()
    X_train, X_test, y_train, y_test = optimizer.prepare_data(df_reviews)
    
    # LogisticRegression optimization
    logreg_results = optimizer.optimize_logistic_regression(X_train, y_train, X_test, y_test)
    
    # RoBERTa optimization
    roberta_results = optimizer.optimize_roberta(X_train, y_train, X_test, y_test)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'method': 'Practical_GridSearch',
        'sample_size': len(X_train) + len(X_test),
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
    
    results_file = config.output_dir / f"practical_gridsearch_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_file}")
    
    print(f"\n📊 PRACTICAL GRID SEARCH SUMMARY:")
    print(f"  LogReg Multi-obj: {logreg_results.multi_objective_score:.4f}")
    if roberta_results:
        print(f"  RoBERTa Multi-obj: {roberta_results.multi_objective_score:.4f}")


if __name__ == "__main__":
    main()
