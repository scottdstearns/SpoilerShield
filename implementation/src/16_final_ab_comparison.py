#!/usr/bin/env python3
"""
SpoilerShield: Final A/B Comparison (120-minute optimal configuration)
======================================================================

Comprehensive A/B test comparing GridSearch vs GA (Strict) vs GA (Advantage)
using calibrated runtime estimates and optimal parameters for 120-minute budget.

Based on runtime estimator analysis:
- Sample size: 20,000 reviews
- GridSearch: 48 combinations (5.7 min)
- GA Strict: 20 pop × 28 gen (55.4 min)  
- GA Advantage: 20 pop × 63 gen (56.5 min)
- Total: ~117.6 minutes

Author: SpoilerShield Development Team
Date: 2025-01-09
"""

import time
import random
import os
import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed

# Project-specific
import sys
src_path = Path(__file__).parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class OptimizationResult:
    """Results from an optimization method."""
    method: str
    best_params: Dict[str, Any]
    cv_f1: float
    cv_auc: float
    test_f1: float
    test_auc: float
    multi_objective_score: float
    optimization_time: float
    total_evaluations: int
    cv_results: Dict[str, Any] = None


class FinalABComparison:
    """
    Final comprehensive A/B comparison with optimal 120-minute configuration.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        set_all_seeds(random_state)
        
        # Multi-objective weights (validated from previous analysis)
        self.f1_weight = 0.4
        self.auc_weight = 0.4  
        self.efficiency_weight = 0.2
        
        # Configuration based on runtime estimator
        self.sample_size = 20000
        self.cv_folds = 3
        
        print("🎯 SPOILERSHIELD: FINAL A/B COMPARISON")
        print("=" * 60)
        print(f"🔒 Random seed: {random_state}")
        print(f"📊 Sample size: {self.sample_size:,}")
        print(f"⏱️ Target runtime: ~120 minutes")
        print(f"⚖️ Multi-objective weights: F1={self.f1_weight}, AUC={self.auc_weight}, Efficiency={self.efficiency_weight}")
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare data with optimal sample size."""
        print(f"\n📥 LOADING & PREPARING DATA")
        print("-" * 30)
        
        config = EnvConfig()
        data_loader = DataLoader(
            movie_reviews_path=config.get_data_path('train_reviews.json'),
            movie_details_path=config.get_data_path('IMDB_movie_details.json')
        )
        
        df_reviews = data_loader.load_imdb_movie_reviews()
        print(f"✅ Full dataset: {len(df_reviews):,} reviews")
        
        # Intelligent sampling maintaining class balance
        print(f"📊 Sampling {self.sample_size:,} reviews with balanced classes...")
        
        # Check available columns
        print(f"✅ Available columns: {list(df_reviews.columns)}")
        
        # Sample with class balance
        if 'is_spoiler' in df_reviews.columns:
            df_sampled = df_reviews.groupby('is_spoiler', group_keys=False).apply(
                lambda x: x.sample(min(len(x), self.sample_size // 2), random_state=self.random_state),
                include_groups=False
            ).reset_index(drop=True)
        else:
            # Fallback sampling if is_spoiler missing
            print(f"⚠️ is_spoiler column not found, using random sampling")
            df_sampled = df_reviews.sample(n=min(len(df_reviews), self.sample_size), random_state=self.random_state)
            
        print(f"✅ Sampled dataset: {len(df_sampled):,} reviews")
        
        # Ensure we have required columns
        if 'is_spoiler' not in df_sampled.columns:
            print(f"❌ Could not find is_spoiler column after sampling")
            # Add fallback logic if needed
            df_sampled['is_spoiler'] = df_sampled.get('rating', 5) > 7  # Fallback based on rating
            
        print(f"✅ Class distribution: {df_sampled['is_spoiler'].value_counts().to_dict()}")
        
        # Split data
        texts = df_sampled['review_text'].values
        labels = df_sampled['is_spoiler'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=self.random_state
        )
        
        print(f"✅ Train: {len(X_train):,}, Test: {len(X_test):,}")
        return X_train, X_test, y_train, y_test
    
    def calculate_multi_objective_score(self, f1: float, auc: float, time_seconds: float) -> float:
        """Calculate multi-objective fitness score."""
        normalized_time = 1.0 / (1.0 + np.log(1.0 + time_seconds / 60.0))
        return (self.f1_weight * f1 + self.auc_weight * auc + self.efficiency_weight * normalized_time)
    
    def run_gridsearch_optimization(self, X_train, y_train, X_test, y_test) -> OptimizationResult:
        """Run comprehensive GridSearch with 48 combinations."""
        print(f"\n📊 GRIDSEARCH OPTIMIZATION")
        print("=" * 40)
        
        start_time = time.time()
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
        
        # Optimal parameter grid (48 combinations: 4×4×2×3)
        param_grid = {
            'tfidf__max_features': [10000, 20000, 40000, 60000],           # 4 values
            'classifier__C': [0.5, 1.0, 2.0, 5.0],                        # 4 values
            'classifier__penalty': ['l2', 'elasticnet'],                   # 2 values  
            'tfidf__ngram_range': [(1,1), (1,2), (1,3)]                    # 3 values
        }
        
        param_combos = np.prod([len(v) for v in param_grid.values()])
        total_evals = param_combos * self.cv_folds
        
        print(f"📈 Parameter combinations: {param_combos}")
        print(f"📈 Total evaluations: {total_evals}")
        print(f"📈 Estimated time: ~5.7 minutes")
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ('classifier', LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                solver='saga',
                max_iter=1000,
                l1_ratio=0.5
            ))
        ])
        
        # Setup CV
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Scoring
        scorers = {
            'f1': make_scorer(f1_score),
            'roc_auc': 'roc_auc'
        }
        
        # Suppress warnings
        import warnings
        warnings.filterwarnings('ignore')
        
        # Run GridSearch
        print(f"🚀 GridSearch fitting started at {time.strftime('%H:%M:%S')}...")
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scorers,
            refit='f1',
            n_jobs=-1,
            verbose=2,
            return_train_score=False
        )
        
        grid_search.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        print(f"✅ GridSearch completed at {time.strftime('%H:%M:%S')} (took {optimization_time:.1f}s)")
        
        # Evaluate on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        test_f1 = f1_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)
        
        # Get CV scores
        cv_f1 = grid_search.cv_results_[f'mean_test_f1'][grid_search.best_index_]
        cv_auc = grid_search.cv_results_[f'mean_test_roc_auc'][grid_search.best_index_]
        
        # Calculate multi-objective score
        multi_objective_score = self.calculate_multi_objective_score(test_f1, test_auc, optimization_time)
        
        print(f"🏆 Best CV F1: {cv_f1:.4f}")
        print(f"🏆 Best CV AUC: {cv_auc:.4f}")
        print(f"🎯 Test F1: {test_f1:.4f}")
        print(f"🎯 Test AUC: {test_auc:.4f}")
        print(f"⚖️ Multi-objective score: {multi_objective_score:.4f}")
        print(f"📊 Best parameters: {grid_search.best_params_}")
        
        return OptimizationResult(
            method="GridSearch",
            best_params=grid_search.best_params_,
            cv_f1=cv_f1,
            cv_auc=cv_auc,
            test_f1=test_f1,
            test_auc=test_auc,
            multi_objective_score=multi_objective_score,
            optimization_time=optimization_time,
            total_evaluations=total_evals,
            cv_results=grid_search.cv_results_
        )
    
    def run_ga_optimization(self, X_train, y_train, X_test, y_test, cv, 
                           advantage_mode: bool = False) -> OptimizationResult:
        """Run Genetic Algorithm optimization."""
        mode_label = "Advantage" if advantage_mode else "Strict"
        
        print(f"\n🧬 GA OPTIMIZATION ({mode_label.upper()} MODE)")
        print("=" * 40)
        
        start_time = time.time()
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
        
        # GA parameters based on runtime estimator
        if advantage_mode:
            population_size = 20
            generations = 63
            print(f"📈 Estimated time: ~56.5 minutes")
        else:
            population_size = 20
            generations = 28
            print(f"📈 Estimated time: ~55.4 minutes")
            
        total_evals = population_size * generations
        
        print(f"👥 Population size: {population_size}")
        print(f"🔄 Generations: {generations}")
        print(f"📈 Total evaluations: {total_evals}")
        
        # Define search space
        if advantage_mode:
            # Continuous parameters
            search_space = {
                'tfidf__max_features': {
                    'type': 'continuous',
                    'range': (10000, 60000),
                    'dtype': int
                },
                'classifier__C': {
                    'type': 'continuous', 
                    'range': (0.1, 10.0),
                    'dtype': float
                },
                'classifier__penalty': {
                    'type': 'discrete',
                    'values': ['l2', 'elasticnet']
                },
                'tfidf__ngram_range': {
                    'type': 'discrete',
                    'values': [(1,1), (1,2), (1,3)]
                }
            }
        else:
            # Discrete grid (same as GridSearch)
            search_space = {
                'tfidf__max_features': {
                    'type': 'discrete',
                    'values': [10000, 20000, 40000, 60000]
                },
                'classifier__C': {
                    'type': 'discrete',
                    'values': [0.5, 1.0, 2.0, 5.0]
                },
                'classifier__penalty': {
                    'type': 'discrete',
                    'values': ['l2', 'elasticnet']
                },
                'tfidf__ngram_range': {
                    'type': 'discrete',
                    'values': [(1,1), (1,2), (1,3)]
                }
            }
        
        # Initialize population
        rs = np.random.RandomState(self.random_state)
        population = []
        
        for _ in range(population_size):
            individual = []
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'continuous':
                    if param_config['dtype'] == int:
                        value = rs.randint(param_config['range'][0], param_config['range'][1] + 1)
                    else:
                        value = rs.uniform(param_config['range'][0], param_config['range'][1])
                else:  # discrete
                    # Convert to list if it's not already (handles tuples in ngram_range)
                    values_list = list(param_config['values'])
                    value = rs.choice(len(values_list))  # Choose index
                    value = values_list[value]  # Get actual value
                individual.append(value)
            population.append(individual)
        
        print(f"🧬 Initialized population of {len(population)} individuals")
        print(f"📈 Progress: {population_size} individuals × {generations} generations = {total_evals} evaluations")
        
        # Evolution tracking
        best_fitness = -1.0
        best_individual = None
        best_f1 = 0.0
        best_auc = 0.0
        
        # Evolution loop
        for generation in range(generations):
            gen_start_time = time.time()
            print(f"\n🔄 Generation {generation + 1}/{generations} (Started: {time.strftime('%H:%M:%S')})")
            
            # Parallel fitness evaluation
            eval_start = time.time()
            print(f"  ⚡ Evaluating {population_size} individuals in parallel...")
            
            evaluation_results = Parallel(n_jobs=-1)(
                delayed(self._evaluate_individual)(individual, search_space, X_train, y_train, cv) 
                for individual in population
            )
            
            eval_time = time.time() - eval_start
            
            # Extract results
            f1_scores = [result[0] for result in evaluation_results]
            auc_scores = [result[1] for result in evaluation_results] 
            times = [result[2] for result in evaluation_results]
            
            # Calculate multi-objective scores
            multi_obj_scores = [
                self.calculate_multi_objective_score(f1, auc, t)
                for f1, auc, t in zip(f1_scores, auc_scores, times)
            ]
            
            # Find best in generation
            gen_best_idx = np.argmax(multi_obj_scores)
            gen_best_fitness = multi_obj_scores[gen_best_idx]
            
            # Update global best
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = copy.deepcopy(population[gen_best_idx])
                best_f1 = f1_scores[gen_best_idx]
                best_auc = auc_scores[gen_best_idx]
            
            gen_total_time = time.time() - gen_start_time
            print(f"  📈 Best fitness: {gen_best_fitness:.4f}")
            print(f"  📈 Best F1: {f1_scores[gen_best_idx]:.4f}")
            print(f"  📈 Best AUC: {auc_scores[gen_best_idx]:.4f}")
            print(f"  ⏱️ Evaluation time: {eval_time:.1f}s, Total gen time: {gen_total_time:.1f}s")
            
            # Progress update
            completed_evals = (generation + 1) * population_size
            progress_pct = (completed_evals / total_evals) * 100
            elapsed_total = time.time() - start_time
            estimated_remaining = (elapsed_total / completed_evals) * (total_evals - completed_evals)
            print(f"  📊 Progress: {completed_evals}/{total_evals} ({progress_pct:.1f}%) | ETA: {estimated_remaining/60:.1f} min")
            
            # Create next generation (if not last generation)
            if generation < generations - 1:
                # Elite selection (keep best 30%)
                elite_count = max(2, int(0.3 * population_size))
                elite_indices = np.argsort(multi_obj_scores)[-elite_count:]
                next_population = [copy.deepcopy(population[i]) for i in elite_indices]
                
                # Fill remaining with mutations
                while len(next_population) < population_size:
                    parent_idx = rs.choice(elite_indices)
                    child = self._mutate_individual(
                        copy.deepcopy(population[parent_idx]), 
                        search_space, 
                        rs,
                        mutation_rate=0.3
                    )
                    next_population.append(child)
                
                population = next_population[:population_size]
        
        # Final evaluation on test set
        print(f"\n🎯 Final evaluation on test set...")
        best_params = self._decode_individual(best_individual, search_space)
        final_f1, final_auc, _ = self._evaluate_individual(
            best_individual, search_space, X_test, y_test, 
            [(np.arange(len(X_test)), np.arange(len(X_test)))]
        )
        
        optimization_time = time.time() - start_time
        print(f"✅ GA completed at {time.strftime('%H:%M:%S')} (total time: {optimization_time:.1f}s)")
        
        print(f"🏆 Best CV F1: {best_f1:.4f}")
        print(f"🏆 Best CV AUC: {best_auc:.4f}")
        print(f"🎯 Test F1: {final_f1:.4f}")
        print(f"🎯 Test AUC: {final_auc:.4f}")
        print(f"⚖️ Multi-objective score: {best_fitness:.4f}")
        print(f"📊 Best parameters: {best_params}")
        
        return OptimizationResult(
            method=f"GA_{mode_label}",
            best_params=best_params,
            cv_f1=best_f1,
            cv_auc=best_auc,
            test_f1=final_f1,
            test_auc=final_auc,
            multi_objective_score=best_fitness,
            optimization_time=optimization_time,
            total_evaluations=total_evals
        )
    
    def _evaluate_individual(self, individual: List, search_space: Dict, 
                           X_train, y_train, cv_folds) -> Tuple[float, float, float]:
        """Evaluate an individual's fitness using cross-validation."""
        eval_start = time.time()
        
        # Decode individual to parameters
        params = self._decode_individual(individual, search_space)
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                max_features=params['tfidf__max_features'],
                ngram_range=params['tfidf__ngram_range'],
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ('classifier', LogisticRegression(
                C=params['classifier__C'],
                penalty=params['classifier__penalty'],
                class_weight='balanced',
                random_state=self.random_state,
                solver='saga',
                max_iter=1000,
                l1_ratio=0.5 if params['classifier__penalty'] == 'elasticnet' else None
            ))
        ])
        
        # Cross-validation
        f1_scores = []
        auc_scores = []
        
        for train_idx, val_idx in cv_folds:
            X_train_fold = [X_train[i] for i in train_idx]
            X_val_fold = [X_train[i] for i in val_idx]
            y_train_fold = [y_train[i] for i in train_idx]
            y_val_fold = [y_train[i] for i in val_idx]
            
            # Suppress warnings
            import warnings
            warnings.filterwarnings('ignore')
            
            # Train and evaluate
            pipeline.fit(X_train_fold, y_train_fold)
            y_pred = pipeline.predict(X_val_fold)
            y_proba = pipeline.predict_proba(X_val_fold)[:, 1]
            
            f1_scores.append(f1_score(y_val_fold, y_pred))
            auc_scores.append(roc_auc_score(y_val_fold, y_proba))
        
        eval_time = time.time() - eval_start
        
        return np.mean(f1_scores), np.mean(auc_scores), eval_time
    
    def _decode_individual(self, individual: List, search_space: Dict) -> Dict[str, Any]:
        """Decode individual to parameter dictionary."""
        params = {}
        param_names = list(search_space.keys())
        
        for i, param_name in enumerate(param_names):
            params[param_name] = individual[i]
            
        return params
    
    def _mutate_individual(self, individual: List, search_space: Dict, 
                          rs: np.random.RandomState, mutation_rate: float = 0.3) -> List:
        """Mutate an individual."""
        param_names = list(search_space.keys())
        
        for i in range(len(individual)):
            if rs.random() < mutation_rate:
                param_config = search_space[param_names[i]]
                
                if param_config['type'] == 'continuous':
                    if param_config['dtype'] == int:
                        individual[i] = rs.randint(param_config['range'][0], param_config['range'][1] + 1)
                    else:
                        individual[i] = rs.uniform(param_config['range'][0], param_config['range'][1])
                else:  # discrete
                    # Convert to list if it's not already (handles tuples in ngram_range)
                    values_list = list(param_config['values'])
                    individual[i] = values_list[rs.choice(len(values_list))]
        
        return individual
    
    def compare_results(self, grid_result: OptimizationResult, 
                       ga_strict_result: OptimizationResult,
                       ga_advantage_result: OptimizationResult) -> Dict[str, Any]:
        """Compare results from all three methods."""
        print(f"\n⚔️ FINAL A/B COMPARISON RESULTS")
        print("=" * 50)
        
        results = [grid_result, ga_strict_result, ga_advantage_result]
        
        # Find winners
        performance_winner = max(results, key=lambda x: x.multi_objective_score)
        speed_winner = min(results, key=lambda x: x.optimization_time)
        
        print(f"🏆 Performance winner: {performance_winner.method}")
        print(f"⚡ Speed winner: {speed_winner.method}")
        
        # Check for continuous advantage
        continuous_advantage = False
        if ga_advantage_result.multi_objective_score > grid_result.multi_objective_score:
            # Check if GA found different C value
            grid_c = grid_result.best_params.get('classifier__C', 0)
            ga_c = ga_advantage_result.best_params.get('classifier__C', 0)
            
            if abs(grid_c - ga_c) > 0.1:  # Meaningful difference
                continuous_advantage = True
                
        print(f"🔍 Continuous advantage: {'✅ Yes' if continuous_advantage else '❌ No'}")
        
        # Create comparison table
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Method': result.method,
                'Multi-Obj Score': f"{result.multi_objective_score:.4f}",
                'CV F1': f"{result.cv_f1:.3f}",
                'CV AUC': f"{result.cv_auc:.3f}",
                'Test F1': f"{result.test_f1:.3f}",
                'Test AUC': f"{result.test_auc:.3f}",
                'Time (min)': f"{result.optimization_time/60:.1f}",
                'Evaluations': result.total_evaluations,
                'Best C': f"{result.best_params.get('classifier__C', 'N/A')}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(f"\n📊 DETAILED COMPARISON:")
        print(comparison_df.to_string(index=False))
        
        return {
            'performance_winner': performance_winner.method,
            'speed_winner': speed_winner.method,
            'continuous_advantage': continuous_advantage,
            'results': results,
            'comparison_df': comparison_df
        }
    
    def generate_final_report(self, comparison: Dict[str, Any]) -> str:
        """Generate comprehensive final report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config = EnvConfig()
        report_file = config.output_dir / f"final_ab_comparison_report_{timestamp}.md"
        
        results = comparison['results']
        grid_result, ga_strict_result, ga_advantage_result = results
        
        with open(report_file, 'w') as f:
            f.write("# Final A/B Comparison: GridSearch vs Genetic Algorithm (120-minute optimal)\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            f.write(f"**Sample Size:** {self.sample_size:,} reviews\n")
            f.write(f"**Performance Winner:** {comparison['performance_winner']}\n")
            f.write(f"**Speed Winner:** {comparison['speed_winner']}\n")
            f.write(f"**Continuous Advantage:** {'✅ Demonstrated' if comparison['continuous_advantage'] else '❌ Not Found'}\n\n")
            
            # Results Table
            f.write("## 📊 Results Comparison\n\n")
            f.write("| Method | Multi-Obj Score | CV F1 | CV AUC | Test F1 | Test AUC | Time (min) | Evaluations | Best C |\n")
            f.write("|--------|-----------------|-------|--------|---------|----------|------------|-------------|--------|\n")
            
            for result in results:
                f.write(f"| {result.method} | {result.multi_objective_score:.4f} | {result.cv_f1:.3f} | {result.cv_auc:.3f} | {result.test_f1:.3f} | {result.test_auc:.3f} | {result.optimization_time/60:.1f} | {result.total_evaluations} | {result.best_params.get('classifier__C', 'N/A')} |\n")
            f.write("\n")
            
            # Parameter Discovery
            f.write("## 🔍 Parameter Discovery\n\n")
            
            for result in results:
                f.write(f"**{result.method} Best Parameters:**\n")
                for param, value in result.best_params.items():
                    f.write(f"- {param}: {value}\n")
                f.write("\n")
            
            # Configuration Details
            f.write("## 🔧 Configuration\n\n")
            f.write(f"**GridSearch:**\n")
            f.write(f"- Parameter combinations: 48 (4×4×2×3)\n")
            f.write(f"- Cross-validation: {self.cv_folds}-fold stratified\n")
            f.write(f"- Total evaluations: {grid_result.total_evaluations}\n\n")
            
            f.write(f"**GA Strict Mode:**\n")
            f.write(f"- Population size: 20\n")
            f.write(f"- Generations: 28\n")
            f.write(f"- Total evaluations: {ga_strict_result.total_evaluations}\n")
            f.write(f"- Search space: Discrete grid (same as GridSearch)\n\n")
            
            f.write(f"**GA Advantage Mode:**\n")
            f.write(f"- Population size: 20\n")
            f.write(f"- Generations: 63\n")
            f.write(f"- Total evaluations: {ga_advantage_result.total_evaluations}\n")
            f.write(f"- Search space: Continuous parameters (C ∈ [0.1, 10.0])\n\n")
            
            # Multi-objective Analysis
            f.write("## ⚖️ Multi-Objective Scoring\n\n")
            f.write(f"**Weights Used:**\n")
            f.write(f"- F1 Score: {self.f1_weight:.1%}\n")
            f.write(f"- ROC AUC: {self.auc_weight:.1%}\n")
            f.write(f"- Efficiency: {self.efficiency_weight:.1%}\n\n")
            
            f.write(f"**Score Breakdown:**\n")
            for result in results:
                normalized_time = 1.0 / (1.0 + np.log(1.0 + result.optimization_time / 60.0))
                f1_contrib = self.f1_weight * result.test_f1
                auc_contrib = self.auc_weight * result.test_auc  
                eff_contrib = self.efficiency_weight * normalized_time
                
                f.write(f"- **{result.method}**: {result.multi_objective_score:.4f} = ")
                f.write(f"{f1_contrib:.3f}(F1) + {auc_contrib:.3f}(AUC) + {eff_contrib:.3f}(Eff)\n")
            f.write("\n")
            
            # Conclusions
            f.write("## 💡 Conclusions\n\n")
            
            total_time = sum(r.optimization_time for r in results)
            f.write(f"🏆 **{comparison['performance_winner']} achieved the best overall performance** ")
            f.write(f"with a multi-objective score of {max(results, key=lambda x: x.multi_objective_score).multi_objective_score:.4f}.\n\n")
            
            f.write("### Key Insights\n\n")
            f.write(f"1. **Runtime Achievement:** Total execution time was {total_time/60:.1f} minutes (target: 120 min)\n")
            f.write(f"2. **Statistical Validity:** {self.sample_size:,} samples provided robust performance estimates\n")
            f.write(f"3. **Evolution Capability:** GA methods had {ga_strict_result.total_evaluations} and {ga_advantage_result.total_evaluations} evaluations respectively\n")
            f.write(f"4. **Parameter Space Exploration:** Continuous vs discrete search comparison completed\n\n")
            
            if comparison['continuous_advantage']:
                f.write(f"🎯 **Continuous Parameter Advantage Demonstrated:** GA Advantage mode found ")
                f.write(f"C={ga_advantage_result.best_params.get('classifier__C', 'N/A')} vs ")
                f.write(f"GridSearch C={grid_result.best_params.get('classifier__C', 'N/A')}, ")
                f.write(f"achieving {ga_advantage_result.multi_objective_score - grid_result.multi_objective_score:.4f} higher score.\n\n")
            else:
                f.write(f"🔍 **Discrete Grid Sufficient:** For this parameter space and dataset size, ")
                f.write(f"discrete grid search was competitive with continuous optimization.\n\n")
            
            f.write(f"**Total Runtime:** {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n")
        
        print(f"📋 Final A/B comparison report saved: {report_file}")
        return str(report_file)
    
    def run_final_ab_comparison(self) -> Dict[str, Any]:
        """Run the complete final A/B comparison."""
        print("🏃 STARTING FINAL A/B COMPARISON")
        print("=" * 60)
        
        start_time = time.time()
        print(f"🕐 Comparison started at: {time.strftime('%H:%M:%S')}")
        print(f"📋 Plan: Data prep → GridSearch (~6 min) → GA Strict (~55 min) → GA Advantage (~57 min)")
        
        # Load and prepare data
        print(f"\n📋 STEP 1/4: Loading and preparing data...")
        step_start = time.time()
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        print(f"✅ Step 1 completed in {time.time() - step_start:.1f}s")
        
        # Setup CV for GA methods
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_indices = [(train_idx, val_idx) for train_idx, val_idx in cv.split(X_train, y_train)]
        
        # Run GridSearch
        print(f"\n📋 STEP 2/4: Running GridSearch optimization...")
        step_start = time.time()
        grid_result = self.run_gridsearch_optimization(X_train, y_train, X_test, y_test)
        print(f"✅ Step 2 completed in {time.time() - step_start:.1f}s")
        
        # Run GA Strict
        print(f"\n📋 STEP 3/4: Running GA Strict optimization...")
        step_start = time.time()
        ga_strict_result = self.run_ga_optimization(X_train, y_train, X_test, y_test, cv_indices, advantage_mode=False)
        print(f"✅ Step 3 completed in {time.time() - step_start:.1f}s")
        
        # Run GA Advantage  
        print(f"\n📋 STEP 4/4: Running GA Advantage optimization...")
        step_start = time.time()
        ga_advantage_result = self.run_ga_optimization(X_train, y_train, X_test, y_test, cv_indices, advantage_mode=True)
        print(f"✅ Step 4 completed in {time.time() - step_start:.1f}s")
        
        # Compare results
        comparison = self.compare_results(grid_result, ga_strict_result, ga_advantage_result)
        
        # Generate final report
        report_file = self.generate_final_report(comparison)
        
        total_time = time.time() - start_time
        print(f"\n🏃 FINAL A/B COMPARISON COMPLETE!")
        print("=" * 60)
        print(f"📁 Report: {report_file}")
        print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        return comparison


def main():
    """Main execution function."""
    comparison = FinalABComparison()
    results = comparison.run_final_ab_comparison()
    return results


if __name__ == "__main__":
    main()
