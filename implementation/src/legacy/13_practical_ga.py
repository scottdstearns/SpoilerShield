#!/usr/bin/env python3
"""
Practical Genetic Algorithm with optimized population and generations for reasonable runtime.
"""

import time
import copy
import random
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from dataclasses import dataclass

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
class PracticalGAResult:
    model_type: str
    best_params: Dict[str, Any]
    best_cv_f1: float
    best_cv_auc: float
    best_multi_objective: float
    test_f1: float
    test_auc: float
    optimization_time: float
    total_evals: int
    ga_config: Dict[str, Any]


class PracticalGeneticOptimizer:
    """Practical GA with balanced runtime vs exploration capability."""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1, sample_size: int = 20000):
        self.random_state = random_state
        self.sample_size = sample_size
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        
        set_all_seeds(random_state)
        self.rs = np.random.RandomState(random_state)
        
        # Practical GA parameters
        self.population_size = 20        # Reduced from 40 for practical runtime
        self.generations = 8             # Reduced from 15 for practical runtime
        self.mutation_rate = 0.25        # Slightly higher for faster exploration
        self.crossover_rate = 0.8
        self.elite_size = 4              # Reduced proportionally
        self.tournament_size = 3         # Reduced for faster selection
        
        # Multi-objective weights (same as GridSearch)
        self.f1_weight = 0.4
        self.auc_weight = 0.5
        self.efficiency_weight = 0.1
        
        self._cv_indices = None
        
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        print(f"🧬 PRACTICAL GENETIC ALGORITHM")
        print(f"🔄 Parallel workers: {self.n_jobs}")
        print(f"👥 Population: {self.population_size}")
        print(f"🔄 Generations: {self.generations}")
    
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
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=self.random_state
        )
        
        # Set CV folds (same as GridSearch for fair comparison)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        self._cv_indices = list(skf.split(X_tr, y_tr))
        
        return X_tr, X_te, y_tr, y_te
    
    def logistic_search_space(self, advantage_mode: bool = False):
        """Define LogReg search space matching GridSearch scope."""
        if advantage_mode:
            return {
                'tfidf__max_features': {'type': 'discrete', 'values': [10000, 20000, 40000]},
                'classifier__C': {'type': 'continuous', 'min': 0.5, 'max': 5.0},  # Continuous advantage
                'classifier__penalty': {'type': 'categorical', 'values': ['l2', 'elasticnet']},
                'tfidf__ngram_range': {'type': 'categorical', 'values': [(1,1), (1,2)]},
            }
        else:
            return {
                'tfidf__max_features': {'type': 'discrete', 'values': [10000, 20000, 40000]},
                'classifier__C': {'type': 'discrete', 'values': [0.5, 1.0, 2.0, 5.0]},
                'classifier__penalty': {'type': 'categorical', 'values': ['l2', 'elasticnet']},
                'tfidf__ngram_range': {'type': 'categorical', 'values': [(1,1), (1,2)]},
            }
    
    def roberta_search_space(self, advantage_mode: bool = False):
        """Define RoBERTa search space matching GridSearch scope."""
        if advantage_mode:
            return {
                'model_name': {'type': 'categorical', 'values': ['roberta-base', 'roberta-large']},
                'learning_rate': {'type': 'continuous', 'min': 3e-5, 'max': 5e-5},  # Continuous advantage
                'max_length': {'type': 'discrete', 'values': [256, 512]},
            }
        else:
            return {
                'model_name': {'type': 'categorical', 'values': ['roberta-base', 'roberta-large']},
                'learning_rate': {'type': 'discrete', 'values': [3e-5, 5e-5]},
                'max_length': {'type': 'discrete', 'values': [256, 512]},
            }
    
    def calculate_multi_objective_score(self, f1: float, auc: float, time_seconds: float) -> float:
        """Calculate multi-objective fitness score."""
        normalized_time = 1.0 / (1.0 + np.log(1.0 + time_seconds / 60.0))
        return (self.f1_weight * f1 + self.auc_weight * auc + self.efficiency_weight * normalized_time)
    
    def create_individual(self, search_space: Dict[str, Dict]) -> List[Any]:
        """Create a random individual."""
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
                index = self.rs.randint(0, len(param_config['values']))
                individual.append(index)
        
        return individual
    
    def decode_individual(self, individual: List[Any], search_space: Dict[str, Dict]) -> Dict[str, Any]:
        """Decode individual to parameters."""
        params = {}
        
        for i, (param_name, param_config) in enumerate(search_space.items()):
            gene_value = individual[i]
            param_type = param_config['type']
            
            if param_type == 'continuous':
                params[param_name] = float(gene_value)
            elif param_type == 'discrete':
                params[param_name] = gene_value
            elif param_type == 'categorical':
                cat_index = int(gene_value) % len(param_config['values'])
                params[param_name] = param_config['values'][cat_index]
        
        return params
    
    def evaluate_individual_logistic(self, individual: List[Any], search_space: Dict[str, Dict],
                                   X_train, y_train) -> Tuple[float, float, float]:
        """Evaluate individual for LogReg."""
        try:
            params = self.decode_individual(individual, search_space)
            
            start_time = time.time()
            
            # Create pipeline with fixed parameters
            tfidf_params = {
                'stop_words': 'english',
                'lowercase': True,
                'strip_accents': 'unicode',
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True
            }
            
            classifier_params = {
                'random_state': self.random_state,
                'solver': 'saga',
                'class_weight': 'balanced',
                'max_iter': 2000,
                'n_jobs': 1
            }
            
            # Extract parameters from individual
            for key, value in params.items():
                if key.startswith('tfidf__'):
                    tfidf_key = key.replace('tfidf__', '')
                    tfidf_params[tfidf_key] = value
                elif key.startswith('classifier__'):
                    classifier_key = key.replace('classifier__', '')
                    classifier_params[classifier_key] = value
            
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
                
                pipeline.fit(X_fold_train, y_fold_train)
                y_pred = pipeline.predict(X_fold_val)
                y_proba = pipeline.predict_proba(X_fold_val)[:, 1]
                
                f1_scores.append(f1_score(y_fold_val, y_pred))
                auc_scores.append(roc_auc_score(y_fold_val, y_proba))
            
            training_time = time.time() - start_time
            return np.mean(f1_scores), np.mean(auc_scores), training_time
            
        except Exception as e:
            return 0.0, 0.5, 999.0
    
    def tournament_selection(self, population: List[List[Any]], fitness_scores: List[float]) -> List[Any]:
        """Tournament selection."""
        tournament_indices = self.rs.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return copy.deepcopy(population[winner_idx])
    
    def uniform_crossover(self, parent1: List[Any], parent2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Uniform crossover."""
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
        """Adaptive mutation."""
        adaptive_rate = self.mutation_rate * (1.0 - generation / max(1, self.generations))
        
        mutated = copy.deepcopy(individual)
        
        for i, (param_name, param_config) in enumerate(search_space.items()):
            if self.rs.random() < adaptive_rate:
                param_type = param_config['type']
                
                if param_type == 'continuous':
                    std = (param_config['max'] - param_config['min']) * 0.1 * (1.0 - generation / max(1, self.generations))
                    noise = self.rs.normal(0, std)
                    mutated[i] = np.clip(mutated[i] + noise, param_config['min'], param_config['max'])
                elif param_type == 'discrete':
                    mutated[i] = self.rs.choice(param_config['values'])
                elif param_type == 'categorical':
                    mutated[i] = self.rs.randint(0, len(param_config['values']))
        
        return mutated
    
    def optimize_genetic_algorithm(self, model_type: str, advantage_mode: bool,
                                 X_train, y_train, X_test, y_test) -> PracticalGAResult:
        """Run practical GA optimization."""
        print(f"\n🧬 PRACTICAL GA: {model_type.upper()} ({'ADVANTAGE' if advantage_mode else 'STRICT'})")
        print("=" * 60)
        
        start_time = time.time()
        
        # Define search space
        if model_type == 'logistic':
            search_space = self.logistic_search_space(advantage_mode)
            eval_func = self.evaluate_individual_logistic
        else:
            raise ValueError(f"Only logistic supported in practical version")
        
        print(f"📊 Search space: {len(search_space)} parameters")
        print(f"📊 Budget: {self.population_size * self.generations} evaluations")
        
        # Initialize population
        population = [self.create_individual(search_space) for _ in range(self.population_size)]
        
        # Evolution tracking
        best_fitness = -1.0
        best_individual = None
        best_f1 = -1.0
        best_auc = -1.0
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\n🔄 Generation {generation + 1}/{self.generations}")
            
            # Parallel fitness evaluation
            eval_start = time.time()
            
            # Suppress warnings during parallel evaluation
            import warnings
            warnings.filterwarnings('ignore')
            
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
                
                multi_obj = self.calculate_multi_objective_score(f1, auc, train_time)
                multi_obj_scores.append(multi_obj)
            
            # Track best
            gen_best_idx = np.argmax(multi_obj_scores)
            gen_best_fitness = multi_obj_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = copy.deepcopy(population[gen_best_idx])
                best_f1 = f1_scores[gen_best_idx]
                best_auc = auc_scores[gen_best_idx]
            
            print(f"  📈 Best fitness: {gen_best_fitness:.4f}")
            print(f"  📈 Best F1: {f1_scores[gen_best_idx]:.4f}")
            print(f"  📈 Best AUC: {auc_scores[gen_best_idx]:.4f}")
            print(f"  ⏱️ Evaluation time: {eval_time:.1f}s")
            
            # Create next generation
            if generation < self.generations - 1:
                # Selection and reproduction
                selected = [self.tournament_selection(population, multi_obj_scores) 
                           for _ in range(self.population_size)]
                
                # Elitism
                elite_indices = np.argsort(multi_obj_scores)[-self.elite_size:]
                elite = [copy.deepcopy(population[i]) for i in elite_indices]
                
                # Create new population
                new_population = elite.copy()
                
                while len(new_population) < self.population_size:
                    parent1 = selected[self.rs.randint(0, len(selected))]
                    parent2 = selected[self.rs.randint(0, len(selected))]
                    
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                    
                    child1 = self.adaptive_mutation(child1, search_space, generation)
                    child2 = self.adaptive_mutation(child2, search_space, generation)
                    
                    new_population.extend([child1, child2])
                
                population = new_population[:self.population_size]
        
        # Final evaluation on test set
        best_params = self.decode_individual(best_individual, search_space)
        final_f1, final_auc, _ = eval_func(best_individual, search_space, X_test, y_test)
        
        optimization_time = time.time() - start_time
        total_evals = self.population_size * self.generations
        
        result = PracticalGAResult(
            model_type=f'{model_type}_GA_{"advantage" if advantage_mode else "strict"}',
            best_params=best_params,
            best_cv_f1=best_f1,
            best_cv_auc=best_auc,
            best_multi_objective=best_fitness,
            test_f1=final_f1,
            test_auc=final_auc,
            optimization_time=optimization_time,
            total_evals=total_evals,
            ga_config={
                'population_size': self.population_size,
                'generations': self.generations,
                'advantage_mode': advantage_mode,
                'n_jobs': self.n_jobs
            }
        )
        
        print(f"\n✅ GA optimization completed in {optimization_time:.1f} seconds")
        print(f"🏆 Best multi-objective: {best_fitness:.4f}")
        print(f"🎯 Test F1: {final_f1:.4f}")
        print(f"🎯 Test AUC: {final_auc:.4f}")
        
        return result


def main():
    """Main execution function."""
    print("🧬 SPOILERSHIELD: PRACTICAL GENETIC ALGORITHM")
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
    optimizer = PracticalGeneticOptimizer()
    X_train, X_test, y_train, y_test = optimizer.prepare_data(df_reviews)
    
    # LogReg optimization (both modes)
    logreg_strict = optimizer.optimize_genetic_algorithm('logistic', False, X_train, y_train, X_test, y_test)
    logreg_advantage = optimizer.optimize_genetic_algorithm('logistic', True, X_train, y_train, X_test, y_test)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'method': 'Practical_GeneticAlgorithm',
        'sample_size': len(X_train) + len(X_test),
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
        }
    }
    
    results_file = config.output_dir / f"practical_ga_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_file}")
    
    print(f"\n📊 PRACTICAL GA SUMMARY:")
    print(f"  LogReg Strict: {logreg_strict.best_multi_objective:.4f}")
    print(f"  LogReg Advantage: {logreg_advantage.best_multi_objective:.4f}")


if __name__ == "__main__":
    main()
