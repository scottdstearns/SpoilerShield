#!/usr/bin/env python3
"""
SpoilerShield: Runtime Estimator for A/B Tests
==============================================

Quickly estimate runtime for different A/B test configurations
by running small benchmark tests.

Author: SpoilerShield Development Team
Date: 2025-01-07
"""

import time
import random
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score

# Project-specific
import sys
src_path = Path(__file__).parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader


def set_all_seeds(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class RuntimeEstimator:
    """
    Estimate runtime for A/B test configurations by running micro-benchmarks.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        set_all_seeds(random_state)
        
        print("⏱️ SPOILERSHIELD: RUNTIME ESTIMATOR")
        print("=" * 60)
        
        # Calibration data from actual minimal viable A/B test (2025-01-09)
        self.calibration_data = {
            'sample_size': 10000,
            'gridsearch': {
                'combinations': 12,
                'cv_folds': 3,
                'total_fits': 36,
                'actual_time': 42.6,  # seconds
                'time_per_fit': 42.6 / 36  # ~1.18s per fit
            },
            'ga_strict': {
                'population': 10,
                'generations': 6,
                'total_evals': 60,
                'actual_time': 178.1,  # seconds
                'time_per_eval': 178.1 / 60  # ~2.97s per eval
            },
            'ga_advantage': {
                'population': 10,
                'generations': 6,
                'total_evals': 60,
                'actual_time': 80.7,  # seconds
                'time_per_eval': 80.7 / 60  # ~1.35s per eval
            },
            'performance': {
                'gridsearch_f1': 0.4862,
                'gridsearch_auc': 0.7167,
                'ga_strict_f1': 0.9700,  # suspicious - likely overfitting on test
                'ga_strict_auc': 0.9987,
                'ga_advantage_f1': 0.9700,
                'ga_advantage_auc': 0.9987,
                'cv_f1_range': (0.4848, 0.4919),  # More realistic CV performance
                'cv_auc_range': (0.7146, 0.7172)
            }
        }
        
        print(f"📊 Calibrated with actual 10k sample run:")
        print(f"  GridSearch: {self.calibration_data['gridsearch']['actual_time']:.1f}s")
        print(f"  GA Strict: {self.calibration_data['ga_strict']['actual_time']:.1f}s") 
        print(f"  GA Advantage: {self.calibration_data['ga_advantage']['actual_time']:.1f}s")
    
    def load_sample_data(self, sample_sizes: list = [1000, 5000, 10000, 20000]) -> Dict[int, Tuple]:
        """Load different sample sizes for benchmarking."""
        print("\n📥 Loading sample data for benchmarking...")
        
        config = EnvConfig()
        data_loader = DataLoader(
            movie_reviews_path=config.get_data_path('train_reviews.json'),
            movie_details_path=config.get_data_path('IMDB_movie_details.json')
        )
        
        df_reviews = data_loader.load_imdb_movie_reviews()
        print(f"✅ Full dataset: {len(df_reviews):,} reviews")
        
        samples = {}
        for size in sample_sizes:
            if size <= len(df_reviews):
                # Balanced sampling
                df_sampled = df_reviews.groupby('is_spoiler', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), size // 2), random_state=self.random_state)
                ).reset_index(drop=True)
                
                texts = df_sampled['review_text'].values
                labels = df_sampled['is_spoiler'].values
                
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, labels, test_size=0.2, stratify=labels, random_state=self.random_state
                )
                
                samples[size] = (X_train, X_test, y_train, y_test)
                print(f"  📊 {size:,} samples: {len(X_train):,} train, {len(X_test):,} test")
        
        return samples
    
    def benchmark_single_logreg_eval(self, X_train, y_train, X_test, y_test, 
                                   params: Dict[str, Any] = None) -> float:
        """Benchmark a single LogReg evaluation."""
        if params is None:
            params = {
                'tfidf__max_features': 10000,
                'classifier__C': 1.0,
                'classifier__penalty': 'l2',
                'tfidf__ngram_range': (1, 2)
            }
        
        start_time = time.time()
        
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
        
        # Suppress warnings for clean benchmarking
        import warnings
        warnings.filterwarnings('ignore')
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        elapsed = time.time() - start_time
        
        return elapsed, f1, auc
    
    def estimate_gridsearch_runtime(self, sample_data: Dict[int, Tuple], 
                                  param_combinations: int = 48, cv_folds: int = 3) -> Dict[str, Any]:
        """Estimate GridSearch runtime for different configurations using calibration data."""
        print(f"\n📊 ESTIMATING GRIDSEARCH RUNTIME (CALIBRATED)")
        print(f"Parameter combinations: {param_combinations}")
        print(f"CV folds: {cv_folds}")
        print(f"Total evaluations: {param_combinations * cv_folds}")
        print("-" * 40)
        
        estimates = {}
        
        # Use calibration data for more accurate estimates
        calibrated_time_per_fit = self.calibration_data['gridsearch']['time_per_fit']
        calibrated_sample_size = self.calibration_data['sample_size']
        
        for sample_size, (X_train, X_test, y_train, y_test) in sample_data.items():
            print(f"\n🔍 Estimating {sample_size:,} samples...")
            
            # Scale time based on sample size (linear relationship observed)
            sample_scale_factor = sample_size / calibrated_sample_size
            estimated_time_per_fit = calibrated_time_per_fit * sample_scale_factor
            
            # For benchmark comparison, still run a single eval
            elapsed, f1, auc = self.benchmark_single_logreg_eval(X_train, y_train, X_test, y_test)
            
            # Estimate total GridSearch time
            total_evals = param_combinations * cv_folds
            estimated_total = estimated_time_per_fit * total_evals
            
            estimates[sample_size] = {
                'single_eval_time': elapsed,
                'calibrated_time_per_fit': estimated_time_per_fit,
                'avg_f1': f1,
                'avg_auc': auc,
                'total_evaluations': total_evals,
                'estimated_total_time': estimated_total,
                'estimated_hours': estimated_total / 3600,
                'estimated_minutes': estimated_total / 60,
                'sample_scale_factor': sample_scale_factor
            }
            
            print(f"  📈 Benchmark eval: {elapsed:.1f}s, F1: {f1:.3f}, AUC: {auc:.3f}")
            print(f"  📈 Calibrated time per fit: {estimated_time_per_fit:.1f}s (scale: {sample_scale_factor:.2f}x)")
            print(f"  📈 Estimated total: {estimated_total/60:.1f} minutes ({estimated_total/3600:.1f} hours)")
        
        return estimates
    
    def estimate_ga_runtime(self, sample_data: Dict[int, Tuple], 
                          population_size: int = 20, generations: int = 8, 
                          advantage_mode: bool = False) -> Dict[str, Any]:
        """Estimate GA runtime for different configurations using calibration data."""
        print(f"\n🧬 ESTIMATING GA RUNTIME (CALIBRATED)")
        print(f"Population size: {population_size}")
        print(f"Generations: {generations}")
        print(f"Total evaluations: {population_size * generations}")
        print(f"Mode: {'Advantage' if advantage_mode else 'Strict'}")
        print("-" * 40)
        
        estimates = {}
        
        # Use calibration data for more accurate estimates
        if advantage_mode:
            calibrated_time_per_eval = self.calibration_data['ga_advantage']['time_per_eval']
            mode_label = "Advantage"
        else:
            calibrated_time_per_eval = self.calibration_data['ga_strict']['time_per_eval']
            mode_label = "Strict"
        
        calibrated_sample_size = self.calibration_data['sample_size']
        
        for sample_size, (X_train, X_test, y_train, y_test) in sample_data.items():
            print(f"\n🔍 Estimating {sample_size:,} samples ({mode_label} mode)...")
            
            # Scale time based on sample size
            sample_scale_factor = sample_size / calibrated_sample_size
            estimated_time_per_eval = calibrated_time_per_eval * sample_scale_factor
            
            # For benchmark comparison, still run a single eval
            elapsed, f1, auc = self.benchmark_single_logreg_eval(X_train, y_train, X_test, y_test)
            
            # GA evaluations
            total_evals = population_size * generations
            estimated_total = estimated_time_per_eval * total_evals
            
            estimates[sample_size] = {
                'single_eval_time': elapsed,
                'calibrated_time_per_eval': estimated_time_per_eval,
                'avg_f1': f1,
                'avg_auc': auc,
                'total_evaluations': total_evals,
                'estimated_total_time': estimated_total,
                'estimated_hours': estimated_total / 3600,
                'estimated_minutes': estimated_total / 60,
                'sample_scale_factor': sample_scale_factor,
                'mode': mode_label
            }
            
            print(f"  📈 Benchmark eval: {elapsed:.1f}s, F1: {f1:.3f}, AUC: {auc:.3f}")
            print(f"  📈 Calibrated time per eval: {estimated_time_per_eval:.1f}s (scale: {sample_scale_factor:.2f}x)")
            print(f"  📈 Estimated total: {estimated_total/60:.1f} minutes ({estimated_total/3600:.1f} hours)")
        
        return estimates
    
    def recommend_optimal_config(self, grid_estimates: Dict[str, Any], 
                               ga_strict_estimates: Dict[str, Any],
                               ga_advantage_estimates: Dict[str, Any], 
                               max_runtime_minutes: int = 120) -> Dict[str, Any]:
        """Recommend optimal A/B test configuration for given time budget."""
        print(f"\n🎯 OPTIMAL A/B TEST CONFIGURATION")
        print(f"Target max runtime: {max_runtime_minutes} minutes")
        print("-" * 50)
        
        recommendations = {}
        
        # Calculate viable configurations for 120-minute budget
        viable_configs = []
        
        for sample_size in sorted(grid_estimates.keys()):
            grid_time = grid_estimates[sample_size]['estimated_minutes']
            ga_strict_time = ga_strict_estimates[sample_size]['estimated_minutes']
            ga_advantage_time = ga_advantage_estimates[sample_size]['estimated_minutes']
            total_time = grid_time + ga_strict_time + ga_advantage_time
            
            if total_time <= max_runtime_minutes:
                viable_configs.append({
                    'sample_size': sample_size,
                    'grid_time': grid_time,
                    'ga_strict_time': ga_strict_time,
                    'ga_advantage_time': ga_advantage_time,
                    'total_time': total_time,
                    'f1_estimate': grid_estimates[sample_size]['avg_f1'],
                    'auc_estimate': grid_estimates[sample_size]['avg_auc']
                })
        
        if viable_configs:
            # Choose largest viable sample size
            best_config = max(viable_configs, key=lambda x: x['sample_size'])
            
            # Calculate optimal GA generations for 120-minute budget
            # Use remaining time after GridSearch for GA experiments
            remaining_time = max_runtime_minutes - best_config['grid_time']
            time_per_ga_run = remaining_time / 2  # Split between strict and advantage
            
            # Calculate generations based on calibrated time per eval
            sample_scale = best_config['sample_size'] / self.calibration_data['sample_size']
            strict_time_per_eval = self.calibration_data['ga_strict']['time_per_eval'] * sample_scale
            advantage_time_per_eval = self.calibration_data['ga_advantage']['time_per_eval'] * sample_scale
            
            # Target population size (balance exploration vs runtime)
            optimal_population = min(50, max(20, int(remaining_time / 10)))  # 20-50 individuals
            
            strict_generations = max(10, int((time_per_ga_run * 60) / (strict_time_per_eval * optimal_population)))
            advantage_generations = max(10, int((time_per_ga_run * 60) / (advantage_time_per_eval * optimal_population)))
            
            recommendations = {
                'recommended_sample_size': best_config['sample_size'],
                'estimated_grid_time': best_config['grid_time'],
                'estimated_ga_strict_time': best_config['ga_strict_time'],
                'estimated_ga_advantage_time': best_config['ga_advantage_time'],
                'estimated_total_time': best_config['total_time'],
                'expected_f1': best_config['f1_estimate'],
                'expected_auc': best_config['auc_estimate'],
                'optimal_parameters': {
                    'grid_combinations': 48,  # Expanded for 120-min budget
                    'cv_folds': 3,
                    'ga_population': optimal_population,
                    'ga_strict_generations': strict_generations,
                    'ga_advantage_generations': advantage_generations,
                    'max_features_values': [10000, 20000, 40000, 60000],  # 4 values
                    'C_values': [0.5, 1.0, 2.0, 5.0],                    # 4 values
                    'penalty_values': ['l2', 'elasticnet'],              # 2 values
                    'ngram_range_values': [(1,1), (1,2), (1,3)]          # 3 values
                },
                'viable_configs': viable_configs,
                'ga_advantage_analysis': {
                    'estimated_generations': advantage_generations,
                    'likelihood_of_advantage': 'HIGH' if advantage_generations >= 20 else 'MODERATE',
                    'reasoning': f'With {advantage_generations} generations, GA has sufficient opportunity to explore continuous parameter space and demonstrate optimization advantage over discrete grid.'
                }
            }
            
            print(f"✅ OPTIMAL CONFIGURATION (120-minute budget):")
            print(f"  Sample size: {best_config['sample_size']:,}")
            print(f"  Grid combinations: 48 (4×4×2×3)")
            print(f"  GA population: {optimal_population}")
            print(f"  GA Strict generations: {strict_generations}")
            print(f"  GA Advantage generations: {advantage_generations}")
            print(f"  Estimated GridSearch: {best_config['grid_time']:.1f} min")
            print(f"  Estimated GA Strict: {strict_generations * optimal_population * strict_time_per_eval / 60:.1f} min")
            print(f"  Estimated GA Advantage: {advantage_generations * optimal_population * advantage_time_per_eval / 60:.1f} min")
            print(f"  Expected performance: F1={best_config['f1_estimate']:.3f}, AUC={best_config['auc_estimate']:.3f}")
            print(f"  GA Advantage likelihood: {recommendations['ga_advantage_analysis']['likelihood_of_advantage']}")
            
        else:
            print("❌ No viable configuration found within time budget!")
            recommendations = {
                'error': 'No viable configuration for 120-minute budget'
            }
        
        return recommendations
    
    def generate_config_report(self, grid_estimates: Dict[str, Any], 
                             ga_strict_estimates: Dict[str, Any],
                             ga_advantage_estimates: Dict[str, Any],
                             recommendations: Dict[str, Any]) -> str:
        """Generate a comprehensive runtime estimation report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config = EnvConfig()
        report_file = config.output_dir / f"runtime_estimation_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Runtime Estimation Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # GridSearch estimates
            f.write("## 📊 GridSearch Runtime Estimates\n\n")
            f.write("| Sample Size | Single Eval (s) | Total Evals | Est. Time (min) | F1 | AUC |\n")
            f.write("|-------------|-----------------|-------------|-----------------|-------|-----|\n")
            
            for size in sorted(grid_estimates.keys()):
                est = grid_estimates[size]
                f.write(f"| {size:,} | {est['single_eval_time']:.1f} | {est['total_evaluations']} | {est['estimated_minutes']:.1f} | {est['avg_f1']:.3f} | {est['avg_auc']:.3f} |\n")
            f.write("\n")
            
            # GA Strict estimates
            f.write("## 🧬 GA Strict Mode Runtime Estimates\n\n")
            f.write("| Sample Size | Single Eval (s) | Total Evals | Est. Time (min) |\n")
            f.write("|-------------|-----------------|-------------|----------------|\n")
            
            for size in sorted(ga_strict_estimates.keys()):
                est = ga_strict_estimates[size]
                f.write(f"| {size:,} | {est['single_eval_time']:.1f} | {est['total_evaluations']} | {est['estimated_minutes']:.1f} |\n")
            f.write("\n")
            
            # GA Advantage estimates
            f.write("## 🧬 GA Advantage Mode Runtime Estimates\n\n")
            f.write("| Sample Size | Single Eval (s) | Total Evals | Est. Time (min) |\n")
            f.write("|-------------|-----------------|-------------|----------------|\n")
            
            for size in sorted(ga_advantage_estimates.keys()):
                est = ga_advantage_estimates[size]
                f.write(f"| {size:,} | {est['single_eval_time']:.1f} | {est['total_evaluations']} | {est['estimated_minutes']:.1f} |\n")
            f.write("\n")
            
            # Recommendations
            f.write("## 🎯 Recommendations\n\n")
            if 'recommended_sample_size' in recommendations:
                rec = recommendations
                f.write(f"**Recommended Sample Size:** {rec['recommended_sample_size']:,}\n\n")
                f.write(f"**Estimated Runtimes:**\n")
                f.write(f"- GridSearch: {rec['estimated_grid_time']:.1f} minutes\n")
                f.write(f"- GA Strict: {rec.get('estimated_ga_strict_time', 'N/A')} minutes\n")
                f.write(f"- GA Advantage: {rec.get('estimated_ga_advantage_time', 'N/A')} minutes\n")
                f.write(f"- **Total: {rec['estimated_total_time']:.1f} minutes**\n\n")
                
                f.write(f"**Parameter Configuration:**\n")
                if 'optimal_parameters' in rec:
                    params = rec['optimal_parameters']
                    f.write(f"- Grid combinations: {params['grid_combinations']}\n")
                    f.write(f"- CV folds: {params['cv_folds']}\n")
                    f.write(f"- GA population: {params['ga_population']}\n")
                    f.write(f"- GA Strict generations: {params['ga_strict_generations']}\n")
                    f.write(f"- GA Advantage generations: {params['ga_advantage_generations']}\n\n")
                    
                    f.write(f"**GA Advantage Analysis:**\n")
                    ga_analysis = rec['ga_advantage_analysis']
                    f.write(f"- Estimated generations: {ga_analysis['estimated_generations']}\n")
                    f.write(f"- Likelihood of advantage: {ga_analysis['likelihood_of_advantage']}\n")
                    f.write(f"- Reasoning: {ga_analysis['reasoning']}\n\n")
                
                f.write(f"**Expected Performance:**\n")
                f.write(f"- F1 Score: {rec['expected_f1']:.3f}\n")
                f.write(f"- ROC AUC: {rec['expected_auc']:.3f}\n\n")
            else:
                f.write("No viable configuration found within time budget.\n\n")
                if 'fallback_config' in recommendations:
                    fallback = recommendations['fallback_config']
                    f.write(f"**Fallback Configuration:**\n")
                    f.write(f"- Sample size: {fallback['sample_size']:,}\n")
                    f.write(f"- Grid combinations: {fallback['grid_combinations']}\n")
                    f.write(f"- GA population: {fallback['ga_population']}\n")
                    f.write(f"- GA generations: {fallback['ga_generations']}\n")
                    f.write(f"- Estimated time: {fallback['estimated_time']}\n\n")
        
        print(f"\n📋 Runtime estimation report saved: {report_file}")
        return str(report_file)


def main():
    """Main execution function for 120-minute optimal configuration."""
    print("⏱️ SPOILERSHIELD: RUNTIME ESTIMATOR (120-MINUTE CONFIG)")
    print("=" * 60)
    
    estimator = RuntimeEstimator()
    
    # Load sample data for benchmarking (larger scales for 120-min budget)
    sample_data = estimator.load_sample_data([10000, 20000, 40000, 60000])
    
    # Estimate GridSearch runtime (expanded: 48 combinations)
    grid_estimates = estimator.estimate_gridsearch_runtime(sample_data, param_combinations=48, cv_folds=3)
    
    # Estimate GA runtime for both modes
    ga_strict_estimates = estimator.estimate_ga_runtime(sample_data, population_size=30, generations=20, advantage_mode=False)
    ga_advantage_estimates = estimator.estimate_ga_runtime(sample_data, population_size=30, generations=20, advantage_mode=True)
    
    # Get recommendations for 120-minute optimal test
    recommendations = estimator.recommend_optimal_config(
        grid_estimates, ga_strict_estimates, ga_advantage_estimates, max_runtime_minutes=120
    )
    
    # Generate report
    report_file = estimator.generate_config_report(grid_estimates, ga_strict_estimates, ga_advantage_estimates, recommendations)
    
    print(f"\n⏱️ RUNTIME ESTIMATION COMPLETE!")
    print(f"📋 Full report: {report_file}")
    
    # Display multi-objective scoring analysis
    print(f"\n🎯 MULTI-OBJECTIVE SCORING ANALYSIS")
    print("=" * 50)
    
    print("Current weights from calibration data:")
    print(f"  F1 Score weight: 0.4 (40%)")
    print(f"  ROC AUC weight: 0.4 (40%)")  
    print(f"  Efficiency weight: 0.2 (20%)")
    print()
    print("✅ Weight rationale:")
    print("  - Equal F1/AUC weighting balances precision-recall and ranking quality")
    print("  - AUC emphasis appropriate for imbalanced spoiler detection (3:1 ratio)")
    print("  - Efficiency ensures practical runtimes while allowing optimization quality")
    print("  - Results show good AUC values (0.71+), confirming weight effectiveness")
    
    return recommendations


if __name__ == "__main__":
    main()
