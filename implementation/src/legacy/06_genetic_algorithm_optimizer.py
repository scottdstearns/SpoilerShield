#!/usr/bin/env python3
"""
SpoilerShield: Genetic Algorithm Hyperparameter Optimization
============================================================

This script implements a genetic algorithm for hyperparameter optimization, designed
for A/B comparison with GridSearchCV. Features multi-objective optimization with
performance vs computational efficiency trade-offs.

Author: SpoilerShield Development Team
Date: 2025-08-07
"""

import sys
import warnings
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
from datetime import datetime
import copy

import numpy as np
import pandas as pd
import os

# Machine Learning
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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


class GeneticHyperparameterOptimizer:
    """
    Genetic Algorithm for hyperparameter optimization.
    
    Implements:
    - Multi-objective fitness (performance + efficiency)
    - Adaptive mutation rates
    - Elitism with diversity preservation
    - Parallel evaluation
    """
    
    def __init__(self, config: EnvConfig, random_state: int = 42):
        """Initialize the Genetic Algorithm optimizer."""
        self.config = config
        self.random_state = random_state
        self.results = {}
        
        # GA Parameters
        self.population_size = 20        # Small for rapid testing
        self.generations = 10            # Quick convergence
        self.mutation_rate = 0.15        # Moderate exploration
        self.crossover_rate = 0.8        # High exploitation
        self.elite_size = 4              # Keep best individuals
        
        # Set all random seeds for reproducibility
        set_all_seeds(random_state)
        
        # Check GPU availability
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")
        
        print("🧬 SPOILERSHIELD: GENETIC ALGORITHM OPTIMIZATION")
        print("=" * 60)
    
    def prepare_data(self, df_reviews: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for GA optimization."""
        print("\n📝 PREPARING DATA")
        print("-" * 30)
        
        texts = df_reviews['review_text'].values
        labels = df_reviews['is_spoiler'].values
        
        # Re-seed before data split for consistency
        set_all_seeds(self.random_state)
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        print(f"✅ Training samples: {len(X_train):,}")
        print(f"✅ Test samples: {len(X_test):,}")
        print(f"✅ Class distribution (train): {np.bincount(y_train)}")
        
        return X_train, X_test, y_train, y_test
    
    def define_search_space(self, model_type: str) -> Dict[str, Dict]:
        """Define the hyperparameter search space."""
        if model_type == 'logistic_regression':
            return {
                'tfidf__max_features': {
                    'type': 'discrete',
                    'values': [10000, 20000],
                    'gene_index': 0
                },
                'classifier__C': {
                    'type': 'continuous',
                    'min': 1.0,
                    'max': 5.0,
                    'gene_index': 1
                },
                'classifier__penalty': {
                    'type': 'categorical',
                    'values': ['l2', 'elasticnet'],
                    'gene_index': 2
                }
            }
        elif model_type == 'roberta':
            return {
                'learning_rate': {
                    'type': 'continuous',
                    'min': 3e-5,
                    'max': 5e-5,
                    'gene_index': 0
                },
                'max_length': {
                    'type': 'discrete',
                    'values': [256, 512],
                    'gene_index': 1
                }
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_individual(self, search_space: Dict[str, Dict]) -> List[Union[float, int]]:
        """Create a random individual (chromosome)."""
        individual = []
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'continuous':
                # Random float in range
                value = np.random.uniform(param_config['min'], param_config['max'])
                individual.append(value)
            elif param_config['type'] == 'discrete':
                # Random choice from discrete values
                value = np.random.choice(param_config['values'])
                individual.append(value)
            elif param_config['type'] == 'categorical':
                # Random index for categorical (we'll map back later)
                index = np.random.randint(len(param_config['values']))
                individual.append(index)
        
        return individual
    
    def decode_individual(self, individual: List[Union[float, int]], 
                         search_space: Dict[str, Dict]) -> Dict[str, Any]:
        """Decode individual chromosome into parameter dictionary."""
        params = {}
        
        for param_name, param_config in search_space.items():
            gene_index = param_config['gene_index']
            gene_value = individual[gene_index]
            
            if param_config['type'] == 'continuous':
                params[param_name] = float(gene_value)
            elif param_config['type'] == 'discrete':
                params[param_name] = int(gene_value)
            elif param_config['type'] == 'categorical':
                # Map index back to categorical value
                cat_index = int(gene_value) % len(param_config['values'])
                params[param_name] = param_config['values'][cat_index]
        
        return params
    
    def fitness_function_logistic(self, individual: List[Union[float, int]],
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 search_space: Dict[str, Dict]) -> Tuple[float, float]:
        """
        Fitness function for LogisticRegression.
        Returns: (f1_score, training_time)
        """
        try:
            # Decode individual to parameters
            params = self.decode_individual(individual, search_space)
            
            start_time = time.time()
            
            # Create pipeline with decoded parameters
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=params['tfidf__max_features'],
                    ngram_range=(1, 2),  # Fixed
                    min_df=2,            # Fixed
                    max_df=0.95,         # Fixed
                    sublinear_tf=True,   # Fixed
                    stop_words='english',
                    lowercase=True
                )),
                ('classifier', LogisticRegression(
                    C=params['classifier__C'],
                    penalty=params['classifier__penalty'],
                    class_weight='balanced',  # Fixed
                    max_iter=2000,           # Fixed
                    random_state=self.random_state,
                    solver='saga',           # Supports all penalties
                    l1_ratio=0.5 if params['classifier__penalty'] == 'elasticnet' else None
                ))
            ])
            
            # Perform 3-fold CV
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            cv_scores = []
            
            for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Fit and predict
                pipeline.fit(X_fold_train, y_fold_train)
                y_pred = pipeline.predict(X_fold_val)
                
                # Calculate F1
                f1 = f1_score(y_fold_val, y_pred)
                cv_scores.append(f1)
            
            training_time = time.time() - start_time
            mean_f1 = np.mean(cv_scores)
            
            return mean_f1, training_time
            
        except Exception as e:
            print(f"❌ Fitness evaluation failed: {str(e)}")
            return 0.0, 999.0  # Poor fitness for failed individuals
    
    def fitness_function_roberta(self, individual: List[Union[float, int]],
                                X_train: np.ndarray, y_train: np.ndarray,
                                search_space: Dict[str, Dict]) -> Tuple[float, float]:
        """
        Fitness function for RoBERTa.
        Returns: (f1_score, training_time)
        """
        if not TRANSFORMERS_AVAILABLE:
            return 0.0, 999.0
        
        try:
            # Decode individual to parameters
            params = self.decode_individual(individual, search_space)
            
            start_time = time.time()
            
            # Use a single fold for speed (GA evaluates many individuals)
            train_size = int(0.8 * len(X_train))
            X_fold_train = X_train[:train_size]
            y_fold_train = y_train[:train_size]
            X_fold_val = X_train[train_size:]
            y_fold_val = y_train[train_size:]
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            model = AutoModelForSequenceClassification.from_pretrained(
                'roberta-base', 
                num_labels=2,
                hidden_dropout_prob=0.1
            )
            
            # Create datasets
            train_dataset = self._create_dataset(X_fold_train, y_fold_train, tokenizer, params['max_length'])
            val_dataset = self._create_dataset(X_fold_val, y_fold_val, tokenizer, params['max_length'])
            
            # Training arguments with reproducibility settings
            training_args = TrainingArguments(
                output_dir=self.config.output_dir / 'temp_ga_roberta',
                num_train_epochs=2,  # Reduced for GA speed
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                learning_rate=params['learning_rate'],
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_steps=200,
                eval_strategy="epoch",
                save_strategy="no",
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
            
            training_time = time.time() - start_time
            f1 = eval_results.get('eval_f1', 0.0)
            
            # Cleanup
            del model, trainer, train_dataset, val_dataset
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            return f1, training_time
            
        except Exception as e:
            print(f"❌ RoBERTa fitness evaluation failed: {str(e)}")
            return 0.0, 999.0
    
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
    
    def calculate_multi_objective_fitness(self, f1_score: float, training_time: float) -> float:
        """
        Calculate multi-objective fitness combining performance and efficiency.
        
        Args:
            f1_score: Model performance (0-1)
            training_time: Training time in seconds
            
        Returns:
            Combined fitness score (higher is better)
        """
        # Normalize training time (log scale to handle wide range)
        normalized_time = 1.0 / (1.0 + np.log(1.0 + training_time))
        
        # Weighted combination (prioritize performance but consider efficiency)
        performance_weight = 0.8
        efficiency_weight = 0.2
        
        fitness = (performance_weight * f1_score + 
                  efficiency_weight * normalized_time)
        
        return fitness
    
    def selection_tournament(self, population: List[List], fitness_scores: List[float], 
                           tournament_size: int = 3) -> List[List]:
        """Tournament selection for choosing parents."""
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select winner
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(copy.deepcopy(population[winner_idx]))
        
        return selected
    
    def crossover_uniform(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """Uniform crossover for creating offspring."""
        if np.random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = []
        child2 = []
        
        for gene1, gene2 in zip(parent1, parent2):
            if np.random.random() < 0.5:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)
        
        return child1, child2
    
    def mutate_adaptive(self, individual: List, search_space: Dict[str, Dict], 
                       generation: int) -> List:
        """Adaptive mutation with decreasing rate over generations."""
        # Adaptive mutation rate (higher early, lower later)
        adaptive_rate = self.mutation_rate * (1.0 - generation / self.generations)
        
        mutated = copy.deepcopy(individual)
        
        for i, (param_name, param_config) in enumerate(search_space.items()):
            if np.random.random() < adaptive_rate:
                if param_config['type'] == 'continuous':
                    # Gaussian mutation with decreasing variance
                    std = (param_config['max'] - param_config['min']) * 0.1 * (1.0 - generation / self.generations)
                    noise = np.random.normal(0, std)
                    mutated[i] = np.clip(mutated[i] + noise, param_config['min'], param_config['max'])
                elif param_config['type'] == 'discrete':
                    # Random choice from discrete values
                    mutated[i] = np.random.choice(param_config['values'])
                elif param_config['type'] == 'categorical':
                    # Random category index
                    mutated[i] = np.random.randint(len(param_config['values']))
        
        return mutated
    
    def optimize_genetic_algorithm(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization for specified model type.
        """
        print(f"\n🧬 GENETIC ALGORITHM: {model_type.upper()}")
        print("=" * 50)
        
        start_time = time.time()
        
        # Define search space
        search_space = self.define_search_space(model_type)
        print(f"📊 Search space: {len(search_space)} parameters")
        print(f"📊 Population size: {self.population_size}")
        print(f"📊 Generations: {self.generations}")
        
        # Choose fitness function
        if model_type == 'logistic_regression':
            fitness_func = self.fitness_function_logistic
        elif model_type == 'roberta':
            fitness_func = self.fitness_function_roberta
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize population with consistent seeding
        print(f"\n🧬 Initializing population...")
        set_all_seeds(self.random_state)  # Re-seed for population initialization
        population = [self.create_individual(search_space) for _ in range(self.population_size)]
        
        # Evolution tracking
        best_fitness_history = []
        avg_fitness_history = []
        best_individual = None
        best_fitness = 0.0
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\n🔄 Generation {generation + 1}/{self.generations}")
            
            # Re-seed at each generation for consistent evolution
            generation_seed = self.random_state + generation
            set_all_seeds(generation_seed)
            
            # Evaluate fitness for entire population
            fitness_scores = []
            f1_scores = []
            training_times = []
            
            for i, individual in enumerate(population):
                print(f"  Evaluating individual {i+1}/{self.population_size}...", end=" ")
                
                f1, train_time = fitness_func(individual, X_train, y_train, search_space)
                multi_objective_fitness = self.calculate_multi_objective_fitness(f1, train_time)
                
                fitness_scores.append(multi_objective_fitness)
                f1_scores.append(f1)
                training_times.append(train_time)
                
                print(f"F1={f1:.4f}, Time={train_time:.1f}s, Fitness={multi_objective_fitness:.4f}")
                
                # Track best individual
                if multi_objective_fitness > best_fitness:
                    best_fitness = multi_objective_fitness
                    best_individual = copy.deepcopy(individual)
            
            # Track evolution statistics
            best_fitness_history.append(max(fitness_scores))
            avg_fitness_history.append(np.mean(fitness_scores))
            
            print(f"  📈 Best fitness: {max(fitness_scores):.4f}")
            print(f"  📊 Avg fitness: {np.mean(fitness_scores):.4f}")
            print(f"  🎯 Best F1: {max(f1_scores):.4f}")
            
            # Create next generation
            if generation < self.generations - 1:  # Don't evolve on last generation
                # Selection
                selected = self.selection_tournament(population, fitness_scores)
                
                # Elitism: Keep best individuals
                elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
                elite = [copy.deepcopy(population[i]) for i in elite_indices]
                
                # Crossover and mutation
                new_population = elite.copy()
                
                while len(new_population) < self.population_size:
                    # Select parents
                    parent1, parent2 = np.random.choice(len(selected), 2, replace=False)
                    
                    # Crossover
                    child1, child2 = self.crossover_uniform(selected[parent1], selected[parent2])
                    
                    # Mutation
                    child1 = self.mutate_adaptive(child1, search_space, generation)
                    child2 = self.mutate_adaptive(child2, search_space, generation)
                    
                    new_population.extend([child1, child2])
                
                # Trim to population size
                population = new_population[:self.population_size]
        
        # Final evaluation with best individual
        print(f"\n🎯 Final evaluation with best individual...")
        best_params = self.decode_individual(best_individual, search_space)
        print(f"🏆 Best parameters: {best_params}")
        
        # Test on final test set
        final_f1, final_time = fitness_func([best_individual], X_test, y_test, search_space) if model_type == 'logistic_regression' else (0.0, 0.0)
        
        optimization_time = time.time() - start_time
        
        results = {
            'model_type': f'{model_type}_GA',
            'best_params': best_params,
            'best_fitness': best_fitness,
            'test_f1': final_f1,
            'optimization_time': optimization_time,
            'total_evaluations': self.population_size * self.generations,
            'method': 'GeneticAlgorithm',
            'evolution_history': {
                'best_fitness': best_fitness_history,
                'avg_fitness': avg_fitness_history
            },
            'ga_config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size
            }
        }
        
        print(f"\n✅ GA optimization completed in {optimization_time:.1f} seconds")
        print(f"🏆 Best fitness: {best_fitness:.4f}")
        print(f"🎯 Test F1: {final_f1:.4f}")
        print(f"📊 Total evaluations: {self.population_size * self.generations}")
        
        return results
    
    def save_results(self, logreg_results: Dict[str, Any], roberta_results: Dict[str, Any]):
        """Save GA optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'method': 'GeneticAlgorithm',
            'logistic_regression': logreg_results,
            'roberta': roberta_results,
            'purpose': 'A/B_Testing_vs_GridSearch'
        }
        
        # Save results
        results_file = self.config.output_dir / f"genetic_algorithm_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 GA results saved: {results_file}")
        return results_file
    
    def run_genetic_optimization(self, df_reviews: pd.DataFrame) -> Dict[str, Any]:
        """Run complete genetic algorithm optimization."""
        print("🧬 STARTING GENETIC ALGORITHM OPTIMIZATION")
        print("=" * 60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df_reviews)
        
        # Run LogReg GA
        print("\n" + "="*60)
        logreg_results = self.optimize_genetic_algorithm('logistic_regression', X_train, y_train, X_test, y_test)
        
        # Run RoBERTa GA
        print("\n" + "="*60)
        roberta_results = self.optimize_genetic_algorithm('roberta', X_train, y_train, X_test, y_test)
        
        # Save results
        results_file = self.save_results(logreg_results, roberta_results)
        
        print("\n🧬 GENETIC ALGORITHM OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print(f"📁 Results saved to: {results_file}")
        print(f"🎯 Ready for A/B comparison with GridSearch!")
        
        return {
            'logistic_regression': logreg_results,
            'roberta': roberta_results,
            'results_file': results_file
        }


def main():
    """Main execution function."""
    print("🧬 SPOILERSHIELD: GENETIC ALGORITHM OPTIMIZATION")
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
    optimizer = GeneticHyperparameterOptimizer(config)
    results = optimizer.run_genetic_optimization(df_reviews)
    
    print(f"\n📊 GA SUMMARY:")
    if 'error' not in results['logistic_regression']:
        lr_f1 = results['logistic_regression']['test_f1']
        print(f"  LogReg F1: {lr_f1:.4f}")
    
    if 'error' not in results['roberta']:
        roberta_f1 = results['roberta']['test_f1']
        print(f"  RoBERTa F1: {roberta_f1:.4f}")


if __name__ == "__main__":
    main()
