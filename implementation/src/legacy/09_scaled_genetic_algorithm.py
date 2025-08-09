#!/usr/bin/env python3
"""
SpoilerShield: Scaled Parallelized Genetic Algorithm
===================================================

Features:
- Parallelized fitness evaluation using joblib
- Multi-objective optimization (F1 + AUC + efficiency) 
- Continuous parameter optimization advantage
- LogReg + RoBERTa support
- Estimated runtime: ~20-45 minutes with parallelization

Author: SpoilerShield Development Team  
Date: 2025-01-07
"""

from __future__ import annotations
import time
import copy
import math
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Machine Learning
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score

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
    print("⚠️ Transformers not available. Only LogisticRegression GA will run.")

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
class ScaledGAResult:
    model_type: str
    best_params: Dict[str, Any]
    best_cv_f1: float
    best_cv_auc: float
    best_multi_objective: float
    test_f1: float
    test_auc: float
    optimization_time: float
    total_evals: int
    best_so_far_f1: List[float]
    best_so_far_auc: List[float]
    best_so_far_multi_obj: List[float]
    ga_config: Dict[str, Any]


class ScaledGeneticOptimizer:
    """
    Parallelized Genetic Algorithm for hyperparameter optimization.
    
    Features:
    - Multi-objective fitness (F1 + AUC + efficiency)
    - Parallel fitness evaluation using joblib
    - Continuous parameter optimization
    - Adaptive mutation rates
    - Tournament selection with elitism
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        
        # Set seeds
        set_all_seeds(random_state)
        self.rs = np.random.RandomState(random_state)
        
        # GA Parameters (scaled up for meaningful search)
        self.population_size = 40        # Larger population for parallelization
        self.generations = 15            # More generations for convergence
        self.mutation_rate = 0.2         # Higher for exploration
        self.crossover_rate = 0.8        # High exploitation
        self.elite_size = 8              # Keep more elite individuals
        self.tournament_size = 5         # Larger tournament
        
        # Multi-objective weights (matching GridSearch)
        self.f1_weight = 0.4
        self.auc_weight = 0.5      # Stronger weight on AUC
        self.efficiency_weight = 0.1  # Weaker weight on efficiency
        
        # Tracking
        self._eval_count = 0
        self._cv_indices = None
        
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")
        
        print(f"🧬 SPOILERSHIELD: SCALED GENETIC ALGORITHM")
        print(f"🔄 Parallel workers: {self.n_jobs}")
        print("=" * 60)
    
    def prepare_data(self, df_reviews: pd.DataFrame):
        """Prepare data with consistent splits."""
        texts = df_reviews['review_text'].values
        labels = df_reviews['is_spoiler'].values
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=self.random_state
        )
        
        # Set default CV folds (can be overridden)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        self._cv_indices = list(skf.split(X_tr, y_tr))
        
        return X_tr, X_te, y_tr, y_te
    
    def set_cv_indices(self, cv_indices: List[Tuple[np.ndarray, np.ndarray]]):
        """Set CV indices to match GridSearch."""
        self._cv_indices = cv_indices
    
    def logistic_search_space(self, advantage_mode: bool = False):
        """
        Define LogisticRegression search space.
        
        Args:
            advantage_mode: If True, use continuous parameters for GA advantage
        """
        if advantage_mode:
            # GA Advantage: Continuous parameters
            return {
                'tfidf__max_features': {'type': 'discrete', 'values': [5000, 10000, 20000, 40000]},
                'classifier__C': {'type': 'continuous', 'min': 0.1, 'max': 10.0},
                'classifier__penalty': {'type': 'categorical', 'values': ['l1', 'l2', 'elasticnet']},
                'tfidf__ngram_range': {'type': 'categorical', 'values': [(1,1), (1,2), (1,3), (2,3)]},
            }
        else:
            # Strict A/B: Same discrete space as GridSearch
            return {
                'tfidf__max_features': {'type': 'discrete', 'values': [5000, 10000, 20000, 40000]},
                'classifier__C': {'type': 'discrete', 'values': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
                'classifier__penalty': {'type': 'categorical', 'values': ['l1', 'l2', 'elasticnet']},
                'tfidf__ngram_range': {'type': 'categorical', 'values': [(1,1), (1,2), (1,3), (2,3)]},
            }
    
    def roberta_search_space(self, advantage_mode: bool = False):
        """
        Define RoBERTa search space.
        
        Args:
            advantage_mode: If True, use continuous parameters for GA advantage
        """
        if advantage_mode:
            # GA Advantage: Continuous learning rate
            return {
                'model_name': {'type': 'categorical', 'values': ['roberta-base', 'roberta-large']},
                'learning_rate': {'type': 'continuous', 'min': 1e-5, 'max': 1e-4},
                'num_train_epochs': {'type': 'discrete', 'values': [3, 4]},
                'max_length': {'type': 'discrete', 'values': [256, 512]},
            }
        else:
            # Strict A/B: Same discrete space as GridSearch
            return {
                'model_name': {'type': 'categorical', 'values': ['roberta-base', 'roberta-large']},
                'learning_rate': {'type': 'discrete', 'values': [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4]},
                'num_train_epochs': {'type': 'discrete', 'values': [3, 4]},
                'max_length': {'type': 'discrete', 'values': [256, 512]},
            }
    
    def create_individual(self, search_space: Dict[str, Dict]) -> List[Any]:
        """Create a random individual (chromosome)."""
        individual = []
        
        for param_name, param_config in search_space.items():
            param_type = param_config['type']
            
            if param_type == 'continuous':
                value = self.rs.uniform(param_config['min'], param_config['max'])
                individual.append(value)
            elif param_type == 'discrete':
                value = self.rs.choice(param_config['values'])
                individual.append(value)
            elif param_type == 'categorical':
                # Store index for categorical
                index = self.rs.randint(0, len(param_config['values']))
                individual.append(index)
        
        return individual
    
    def decode_individual(self, individual: List[Any], search_space: Dict[str, Dict]) -> Dict[str, Any]:
        """Decode individual chromosome into parameter dictionary."""
        params = {}
        
        for i, (param_name, param_config) in enumerate(search_space.items()):
            gene_value = individual[i]
            param_type = param_config['type']
            
            if param_type == 'continuous':
                params[param_name] = float(gene_value)
            elif param_type == 'discrete':
                params[param_name] = gene_value
            elif param_type == 'categorical':
                # Map index back to categorical value
                cat_index = int(gene_value) % len(param_config['values'])
                params[param_name] = param_config['values'][cat_index]
        
        return params
    
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
    
    def evaluate_individual_logistic(self, individual: List[Any], search_space: Dict[str, Dict],
                                   X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float, float]:
        """
        Evaluate a single individual for LogisticRegression.
        
        Returns:
            Tuple of (f1_score, auc_score, training_time)
        """
        try:
            # Decode individual to parameters
            params = self.decode_individual(individual, search_space)
            
            start_time = time.time()
            
            # Create pipeline
            tfidf_params = {}
            classifier_params = {}
            
            for key, value in params.items():
                if key.startswith('tfidf__'):
                    tfidf_key = key.replace('tfidf__', '')
                    tfidf_params[tfidf_key] = value
                elif key.startswith('classifier__'):
                    classifier_key = key.replace('classifier__', '')
                    classifier_params[classifier_key] = value
            
            # Set fixed parameters
            tfidf_params.update({
                'stop_words': 'english',
                'lowercase': True,
                'strip_accents': 'unicode',
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True
            })
            
            classifier_params.update({
                'random_state': self.random_state,
                'solver': 'saga',  # Supports all penalties
                'class_weight': 'balanced',
                'max_iter': 3000,
                'n_jobs': 1  # Individual parallelization handled by joblib
            })
            
            # Add l1_ratio for elasticnet
            if classifier_params.get('penalty') == 'elasticnet':
                classifier_params['l1_ratio'] = 0.5
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_params)),
                ('classifier', LogisticRegression(**classifier_params))
            ])
            
            # Cross-validation evaluation
            f1_scores = []
            auc_scores = []
            
            for train_idx, val_idx in self._cv_indices:
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Fit and predict
                pipeline.fit(X_fold_train, y_fold_train)
                y_pred = pipeline.predict(X_fold_val)
                y_proba = pipeline.predict_proba(X_fold_val)[:, 1]
                
                # Calculate metrics
                f1_scores.append(f1_score(y_fold_val, y_pred))
                auc_scores.append(roc_auc_score(y_fold_val, y_proba))
            
            training_time = time.time() - start_time
            mean_f1 = np.mean(f1_scores)
            mean_auc = np.mean(auc_scores)
            
            return mean_f1, mean_auc, training_time
            
        except Exception as e:
            print(f"❌ Individual evaluation failed: {str(e)}")
            return 0.0, 0.5, 999.0  # Poor fitness for failed individuals
    
    def evaluate_individual_roberta(self, individual: List[Any], search_space: Dict[str, Dict],
                                   X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float, float]:
        """
        Evaluate a single individual for RoBERTa.
        
        Returns:
            Tuple of (f1_score, auc_score, training_time)
        """
        if not TRANSFORMERS_AVAILABLE:
            return 0.0, 0.5, 999.0
        
        try:
            # Decode individual to parameters
            params = self.decode_individual(individual, search_space)
            
            start_time = time.time()
            
            # Use single fold for speed (GA evaluates many individuals)
            train_idx, val_idx = self._cv_indices[0]  # Use first fold
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
            model = AutoModelForSequenceClassification.from_pretrained(
                params['model_name'], 
                num_labels=2,
                hidden_dropout_prob=0.1
            )
            
            # Create datasets
            train_dataset = self._create_dataset(X_fold_train, y_fold_train, tokenizer, params['max_length'])
            val_dataset = self._create_dataset(X_fold_val, y_fold_val, tokenizer, params['max_length'])
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=Path.cwd() / 'temp_ga_roberta_scaled',
                num_train_epochs=params['num_train_epochs'],
                per_device_train_batch_size=8,  # Reduced for memory
                per_device_eval_batch_size=8,
                learning_rate=params['learning_rate'],
                weight_decay=0.01,
                warmup_ratio=0.1,
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
            
            training_time = time.time() - start_time
            f1 = eval_results.get('eval_f1', 0.0)
            auc = eval_results.get('eval_roc_auc', 0.5)
            
            # Cleanup
            del model, trainer, train_dataset, val_dataset
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            return f1, auc, training_time
            
        except Exception as e:
            print(f"❌ RoBERTa evaluation failed: {str(e)}")
            return 0.0, 0.5, 999.0
    
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
    
    def tournament_selection(self, population: List[List[Any]], fitness_scores: List[float]) -> List[Any]:
        """Tournament selection for choosing parents."""
        tournament_indices = self.rs.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return copy.deepcopy(population[winner_idx])
    
    def uniform_crossover(self, parent1: List[Any], parent2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Uniform crossover for creating offspring."""
        if self.rs.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1, child2 = [], []
        
        for gene1, gene2 in zip(parent1, parent2):
            if self.rs.random() < 0.5:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)
        
        return child1, child2
    
    def adaptive_mutation(self, individual: List[Any], search_space: Dict[str, Dict], 
                         generation: int) -> List[Any]:
        """Adaptive mutation with decreasing rate over generations."""
        # Adaptive mutation rate (higher early, lower later)
        adaptive_rate = self.mutation_rate * (1.0 - generation / max(1, self.generations))
        
        mutated = copy.deepcopy(individual)
        
        for i, (param_name, param_config) in enumerate(search_space.items()):
            if self.rs.random() < adaptive_rate:
                param_type = param_config['type']
                
                if param_type == 'continuous':
                    # Gaussian mutation with decreasing variance
                    std = (param_config['max'] - param_config['min']) * 0.1 * (1.0 - generation / max(1, self.generations))
                    noise = self.rs.normal(0, std)
                    mutated[i] = np.clip(mutated[i] + noise, param_config['min'], param_config['max'])
                elif param_type == 'discrete':
                    # Random choice from discrete values
                    mutated[i] = self.rs.choice(param_config['values'])
                elif param_type == 'categorical':
                    # Random category index
                    mutated[i] = self.rs.randint(0, len(param_config['values']))
        
        return mutated
    
    def optimize_genetic_algorithm(self, model_type: str, advantage_mode: bool,
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray) -> ScaledGAResult:
        """
        Run scaled genetic algorithm optimization.
        
        Args:
            model_type: 'logistic' or 'roberta'
            advantage_mode: Whether to use continuous parameters
            X_train, y_train, X_test, y_test: Data splits
            
        Returns:
            Optimization results
        """
        print(f"\n🧬 SCALED GA: {model_type.upper()} ({'ADVANTAGE' if advantage_mode else 'STRICT'})")
        print("=" * 60)
        
        start_time = time.time()
        
        # Define search space and evaluation function
        if model_type == 'logistic':
            search_space = self.logistic_search_space(advantage_mode)
            eval_func = self.evaluate_individual_logistic
        elif model_type == 'roberta':
            search_space = self.roberta_search_space(advantage_mode)
            eval_func = self.evaluate_individual_roberta
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"📊 Search space: {len(search_space)} parameters")
        print(f"📊 Population: {self.population_size}")
        print(f"📊 Generations: {self.generations}")
        print(f"📊 Parallel workers: {self.n_jobs}")
        print(f"📊 Advantage mode: {advantage_mode}")
        
        # Initialize population
        print(f"\n🧬 Initializing population...")
        population = [self.create_individual(search_space) for _ in range(self.population_size)]
        
        # Evolution tracking
        best_fitness_history = []
        best_f1_history = []
        best_auc_history = []
        best_multi_obj_history = []
        best_individual = None
        best_fitness = -1.0
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\n🔄 Generation {generation + 1}/{self.generations}")
            
            # Parallel fitness evaluation
            print(f"  ⚡ Evaluating {self.population_size} individuals in parallel...")
            eval_start = time.time()
            
            # Use joblib for parallel evaluation
            evaluation_results = Parallel(n_jobs=self.n_jobs)(
                delayed(eval_func)(individual, search_space, X_train, y_train) 
                for individual in population
            )
            
            eval_time = time.time() - eval_start
            
            # Process results
            f1_scores = []
            auc_scores = []
            training_times = []
            multi_obj_scores = []
            
            for f1, auc, train_time in evaluation_results:
                f1_scores.append(f1)
                auc_scores.append(auc)
                training_times.append(train_time)
                
                # Calculate multi-objective score
                multi_obj = self.calculate_multi_objective_score(f1, auc, train_time)
                multi_obj_scores.append(multi_obj)
            
            # Track best individual
            gen_best_idx = np.argmax(multi_obj_scores)
            gen_best_fitness = multi_obj_scores[gen_best_idx]
            gen_best_f1 = f1_scores[gen_best_idx]
            gen_best_auc = auc_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = copy.deepcopy(population[gen_best_idx])
            
            # Track evolution statistics
            best_fitness_history.append(max(multi_obj_scores))
            best_f1_history.append(max(f1_scores))
            best_auc_history.append(max(auc_scores))
            best_multi_obj_history.append(gen_best_fitness)
            
            # Extend best-so-far for each evaluation
            current_best_f1 = max(best_f1_history)
            current_best_auc = max(best_auc_history)
            current_best_multi_obj = max(best_multi_obj_history)
            
            for _ in range(self.population_size):
                best_f1_history.append(current_best_f1)
                best_auc_history.append(current_best_auc)
                best_multi_obj_history.append(current_best_multi_obj)
            
            print(f"  📈 Best fitness: {gen_best_fitness:.4f}")
            print(f"  📈 Best F1: {gen_best_f1:.4f}")
            print(f"  📈 Best AUC: {gen_best_auc:.4f}")
            print(f"  📈 Mean fitness: {np.mean(multi_obj_scores):.4f}")
            print(f"  ⏱️ Evaluation time: {eval_time:.1f}s")
            
            # Create next generation (if not last generation)
            if generation < self.generations - 1:
                # Selection
                selected = [self.tournament_selection(population, multi_obj_scores) 
                           for _ in range(self.population_size)]
                
                # Elitism: Keep best individuals
                elite_indices = np.argsort(multi_obj_scores)[-self.elite_size:]
                elite = [copy.deepcopy(population[i]) for i in elite_indices]
                
                # Crossover and mutation
                new_population = elite.copy()
                
                while len(new_population) < self.population_size:
                    # Select parents
                    parent1 = selected[self.rs.randint(0, len(selected))]
                    parent2 = selected[self.rs.randint(0, len(selected))]
                    
                    # Crossover
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                    
                    # Mutation
                    child1 = self.adaptive_mutation(child1, search_space, generation)
                    child2 = self.adaptive_mutation(child2, search_space, generation)
                    
                    new_population.extend([child1, child2])
                
                # Trim to population size
                population = new_population[:self.population_size]
        
        # Final evaluation with best individual
        print(f"\n🎯 Final evaluation with best individual...")
        best_params = self.decode_individual(best_individual, search_space)
        print(f"🏆 Best parameters: {best_params}")
        
        # Test on held-out test set
        final_f1, final_auc, final_time = eval_func([best_individual], search_space, X_test, y_test)[0]
        
        optimization_time = time.time() - start_time
        total_evals = self.population_size * self.generations
        
        # Get best CV scores from evolution
        best_cv_f1 = max(f1_scores) if f1_scores else 0.0
        best_cv_auc = max(auc_scores) if auc_scores else 0.0
        
        result = ScaledGAResult(
            model_type=f'{model_type}_GA_{"advantage" if advantage_mode else "strict"}',
            best_params=best_params,
            best_cv_f1=best_cv_f1,
            best_cv_auc=best_cv_auc,
            best_multi_objective=best_fitness,
            test_f1=final_f1,
            test_auc=final_auc,
            optimization_time=optimization_time,
            total_evals=total_evals,
            best_so_far_f1=best_f1_history,
            best_so_far_auc=best_auc_history,
            best_so_far_multi_obj=best_multi_obj_history,
            ga_config={
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size,
                'tournament_size': self.tournament_size,
                'advantage_mode': advantage_mode,
                'n_jobs': self.n_jobs
            }
        )
        
        print(f"\n✅ GA optimization completed in {optimization_time:.1f} seconds")
        print(f"🏆 Best multi-objective: {best_fitness:.4f}")
        print(f"🎯 Test F1: {final_f1:.4f}")
        print(f"🎯 Test AUC: {final_auc:.4f}")
        print(f"📊 Total evaluations: {total_evals}")
        
        return result


def main():
    """Main execution function."""
    print("🧬 SPOILERSHIELD: SCALED GENETIC ALGORITHM")
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
    optimizer = ScaledGeneticOptimizer()
    X_train, X_test, y_train, y_test = optimizer.prepare_data(df_reviews)
    
    # LogisticRegression optimization (both modes)
    logreg_strict = optimizer.optimize_genetic_algorithm('logistic', False, X_train, y_train, X_test, y_test)
    logreg_advantage = optimizer.optimize_genetic_algorithm('logistic', True, X_train, y_train, X_test, y_test)
    
    # RoBERTa optimization (if available)
    if TRANSFORMERS_AVAILABLE:
        roberta_strict = optimizer.optimize_genetic_algorithm('roberta', False, X_train, y_train, X_test, y_test)
        roberta_advantage = optimizer.optimize_genetic_algorithm('roberta', True, X_train, y_train, X_test, y_test)
    else:
        roberta_strict = None
        roberta_advantage = None
    
    # Save results
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'method': 'Scaled_GeneticAlgorithm',
        'logistic_strict': {
            'best_params': logreg_strict.best_params,
            'best_cv_f1': logreg_strict.best_cv_f1,
            'best_cv_auc': logreg_strict.best_cv_auc,
            'best_multi_objective': logreg_strict.best_multi_objective,
            'test_f1': logreg_strict.test_f1,
            'test_auc': logreg_strict.test_auc,
            'optimization_time': logreg_strict.optimization_time,
            'total_evals': logreg_strict.total_evals,
        },
        'logistic_advantage': {
            'best_params': logreg_advantage.best_params,
            'best_cv_f1': logreg_advantage.best_cv_f1,
            'best_cv_auc': logreg_advantage.best_cv_auc,
            'best_multi_objective': logreg_advantage.best_multi_objective,
            'test_f1': logreg_advantage.test_f1,
            'test_auc': logreg_advantage.test_auc,
            'optimization_time': logreg_advantage.optimization_time,
            'total_evals': logreg_advantage.total_evals,
        },
        'roberta_strict': {
            'best_params': roberta_strict.best_params if roberta_strict else {},
            'best_cv_f1': roberta_strict.best_cv_f1 if roberta_strict else 0.0,
            'best_cv_auc': roberta_strict.best_cv_auc if roberta_strict else 0.0,
            'best_multi_objective': roberta_strict.best_multi_objective if roberta_strict else 0.0,
            'test_f1': roberta_strict.test_f1 if roberta_strict else 0.0,
            'test_auc': roberta_strict.test_auc if roberta_strict else 0.0,
            'optimization_time': roberta_strict.optimization_time if roberta_strict else 0.0,
            'total_evals': roberta_strict.total_evals if roberta_strict else 0,
        } if roberta_strict else {'error': 'Transformers not available'},
        'roberta_advantage': {
            'best_params': roberta_advantage.best_params if roberta_advantage else {},
            'best_cv_f1': roberta_advantage.best_cv_f1 if roberta_advantage else 0.0,
            'best_cv_auc': roberta_advantage.best_cv_auc if roberta_advantage else 0.0,
            'best_multi_objective': roberta_advantage.best_multi_objective if roberta_advantage else 0.0,
            'test_f1': roberta_advantage.test_f1 if roberta_advantage else 0.0,
            'test_auc': roberta_advantage.test_auc if roberta_advantage else 0.0,
            'optimization_time': roberta_advantage.optimization_time if roberta_advantage else 0.0,
            'total_evals': roberta_advantage.total_evals if roberta_advantage else 0,
        } if roberta_advantage else {'error': 'Transformers not available'}
    }
    
    results_file = config.output_dir / f"scaled_ga_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_file}")
    
    print(f"\n📊 SCALED GA SUMMARY:")
    print(f"  LogReg Strict: {logreg_strict.best_multi_objective:.4f}")
    print(f"  LogReg Advantage: {logreg_advantage.best_multi_objective:.4f}")
    if roberta_strict:
        print(f"  RoBERTa Strict: {roberta_strict.best_multi_objective:.4f}")
        print(f"  RoBERTa Advantage: {roberta_advantage.best_multi_objective:.4f}")


if __name__ == "__main__":
    main()
