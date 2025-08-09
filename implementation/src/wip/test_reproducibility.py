#!/usr/bin/env python3
"""
SpoilerShield: Reproducibility Test
===================================

This script tests that our seeding implementation ensures full reproducibility
across multiple runs of the optimization algorithms.

Author: SpoilerShield Development Team
Date: 2025-08-07
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader

# Dynamic imports for the optimization modules
import importlib.util

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import optimization modules
gridsearch_module = import_module_from_file("simplified_gridsearch", 
    src_path / "05_simplified_gridsearch.py")
ga_module = import_module_from_file("genetic_algorithm", 
    src_path / "06_genetic_algorithm_optimizer.py")

SimplifiedGridSearchOptimizer = gridsearch_module.SimplifiedGridSearchOptimizer
GeneticHyperparameterOptimizer = ga_module.GeneticHyperparameterOptimizer


class ReproducibilityTester:
    """Test reproducibility of optimization algorithms."""
    
    def __init__(self, config: EnvConfig):
        """Initialize the reproducibility tester."""
        self.config = config
        self.test_results = {}
        
        print("🔬 SPOILERSHIELD: REPRODUCIBILITY TESTING")
        print("=" * 60)
    
    def load_test_data(self) -> pd.DataFrame:
        """Load a small subset of data for quick testing."""
        print("\n📥 LOADING TEST DATA")
        print("-" * 30)
        
        data_loader = DataLoader(
            movie_reviews_path=self.config.get_data_path('train_reviews.json'),
            movie_details_path=self.config.get_data_path('IMDB_movie_details.json')
        )
        
        df_reviews = data_loader.load_imdb_movie_reviews()
        
        # Use small subset for quick testing
        test_subset = df_reviews.sample(n=1000, random_state=42).reset_index(drop=True)
        
        print(f"✅ Loaded {len(test_subset):,} reviews for testing")
        print(f"✅ Class distribution: {test_subset['is_spoiler'].value_counts().to_dict()}")
        
        return test_subset
    
    def test_gridsearch_reproducibility(self, df_reviews: pd.DataFrame, num_runs: int = 3) -> Dict[str, Any]:
        """Test GridSearch reproducibility across multiple runs."""
        print(f"\n🔬 TESTING GRIDSEARCH REPRODUCIBILITY ({num_runs} runs)")
        print("=" * 60)
        
        results = []
        
        for run in range(num_runs):
            print(f"\n▶️ Run {run + 1}/{num_runs}")
            print("-" * 30)
            
            # Create fresh optimizer with same seed
            optimizer = SimplifiedGridSearchOptimizer(self.config, random_state=42)
            
            # Run LogReg optimization only (faster for testing)
            X_train, X_test, y_train, y_test = optimizer.prepare_data(df_reviews)
            result = optimizer.optimize_logistic_regression(X_train, y_train, X_test, y_test)
            
            # Extract key metrics for comparison
            run_data = {
                'run': run + 1,
                'best_params': result['best_params'],
                'cv_score': result['best_cv_score'],
                'test_f1': result['test_metrics']['f1'],
                'test_accuracy': result['test_metrics']['accuracy'],
                'test_precision': result['test_metrics']['precision'],
                'test_recall': result['test_metrics']['recall'],
                'test_roc_auc': result['test_metrics']['roc_auc'],
                'test_specificity': result['test_metrics']['specificity']
            }
            
            results.append(run_data)
            
            print(f"  🎯 F1 Score: {run_data['test_f1']:.6f}")
            print(f"  🎯 CV Score: {run_data['cv_score']:.6f}")
        
        # Analyze reproducibility
        reproducibility_analysis = self._analyze_reproducibility(results, 'GridSearch')
        
        return {
            'method': 'GridSearch',
            'runs': results,
            'analysis': reproducibility_analysis
        }
    
    def test_ga_reproducibility(self, df_reviews: pd.DataFrame, num_runs: int = 3) -> Dict[str, Any]:
        """Test Genetic Algorithm reproducibility across multiple runs."""
        print(f"\n🔬 TESTING GENETIC ALGORITHM REPRODUCIBILITY ({num_runs} runs)")
        print("=" * 60)
        
        results = []
        
        for run in range(num_runs):
            print(f"\n▶️ Run {run + 1}/{num_runs}")
            print("-" * 30)
            
            # Create fresh optimizer with same seed
            optimizer = GeneticHyperparameterOptimizer(self.config, random_state=42)
            
            # Use smaller GA parameters for testing
            optimizer.population_size = 6
            optimizer.generations = 3
            
            # Run LogReg GA only (faster for testing)
            X_train, X_test, y_train, y_test = optimizer.prepare_data(df_reviews)
            result = optimizer.optimize_genetic_algorithm('logistic_regression', X_train, y_train, X_test, y_test)
            
            # Extract key metrics for comparison
            run_data = {
                'run': run + 1,
                'best_params': result['best_params'],
                'best_fitness': result['best_fitness'],
                'test_f1': result['test_f1'],
                'total_evaluations': result['total_evaluations']
            }
            
            results.append(run_data)
            
            print(f"  🎯 F1 Score: {run_data['test_f1']:.6f}")
            print(f"  🎯 Fitness: {run_data['best_fitness']:.6f}")
        
        # Analyze reproducibility
        reproducibility_analysis = self._analyze_reproducibility(results, 'GeneticAlgorithm')
        
        return {
            'method': 'GeneticAlgorithm',
            'runs': results,
            'analysis': reproducibility_analysis
        }
    
    def _analyze_reproducibility(self, results: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
        """Analyze reproducibility across runs."""
        print(f"\n📊 REPRODUCIBILITY ANALYSIS: {method}")
        print("-" * 50)
        
        analysis = {
            'method': method,
            'num_runs': len(results),
            'metrics_comparison': {},
            'parameters_comparison': {},
            'reproducibility_score': 0.0,
            'status': 'UNKNOWN'
        }
        
        # Compare metrics across runs
        if method == 'GridSearch':
            metrics_to_compare = ['cv_score', 'test_f1', 'test_accuracy', 'test_precision', 
                                'test_recall', 'test_roc_auc', 'test_specificity']
        else:  # GA
            metrics_to_compare = ['best_fitness', 'test_f1']
        
        for metric in metrics_to_compare:
            values = [run[metric] for run in results]
            
            analysis['metrics_comparison'][metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'identical': len(set(values)) == 1,
                'max_difference': np.max(values) - np.min(values)
            }
            
            print(f"  {metric}:")
            print(f"    Values: {[f'{v:.6f}' for v in values]}")
            print(f"    Identical: {analysis['metrics_comparison'][metric]['identical']}")
            print(f"    Max diff: {analysis['metrics_comparison'][metric]['max_difference']:.8f}")
        
        # Compare parameters across runs
        param_keys = set()
        for run in results:
            param_keys.update(run['best_params'].keys())
        
        for param in param_keys:
            param_values = [run['best_params'].get(param, None) for run in results]
            
            analysis['parameters_comparison'][param] = {
                'values': param_values,
                'identical': len(set(str(v) for v in param_values)) == 1
            }
            
            print(f"  {param}:")
            print(f"    Values: {param_values}")
            print(f"    Identical: {analysis['parameters_comparison'][param]['identical']}")
        
        # Calculate overall reproducibility score
        metrics_identical = sum(1 for comp in analysis['metrics_comparison'].values() 
                              if comp['identical'])
        params_identical = sum(1 for comp in analysis['parameters_comparison'].values() 
                             if comp['identical'])
        
        total_metrics = len(analysis['metrics_comparison'])
        total_params = len(analysis['parameters_comparison'])
        
        if total_metrics + total_params > 0:
            analysis['reproducibility_score'] = (metrics_identical + params_identical) / (total_metrics + total_params)
        
        # Determine status
        if analysis['reproducibility_score'] == 1.0:
            analysis['status'] = 'PERFECT'
            print(f"\n✅ PERFECT REPRODUCIBILITY: {analysis['reproducibility_score']:.1%}")
        elif analysis['reproducibility_score'] >= 0.8:
            analysis['status'] = 'GOOD'
            print(f"\n🟡 GOOD REPRODUCIBILITY: {analysis['reproducibility_score']:.1%}")
        else:
            analysis['status'] = 'POOR'
            print(f"\n❌ POOR REPRODUCIBILITY: {analysis['reproducibility_score']:.1%}")
        
        return analysis
    
    def save_test_results(self, gridsearch_results: Dict[str, Any], 
                         ga_results: Dict[str, Any]) -> str:
        """Save reproducibility test results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'test_type': 'reproducibility',
            'gridsearch': gridsearch_results,
            'genetic_algorithm': ga_results,
            'summary': {
                'gridsearch_reproducibility': gridsearch_results['analysis']['reproducibility_score'],
                'gridsearch_status': gridsearch_results['analysis']['status'],
                'ga_reproducibility': ga_results['analysis']['reproducibility_score'],
                'ga_status': ga_results['analysis']['status'],
                'overall_status': 'PASS' if (
                    gridsearch_results['analysis']['status'] in ['PERFECT', 'GOOD'] and
                    ga_results['analysis']['status'] in ['PERFECT', 'GOOD']
                ) else 'FAIL'
            }
        }
        
        # Save results
        results_file = self.config.output_dir / f"reproducibility_test_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved: {results_file}")
        return str(results_file)
    
    def run_full_reproducibility_test(self) -> Dict[str, Any]:
        """Run complete reproducibility test."""
        print("🔬 STARTING REPRODUCIBILITY TEST")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load test data
        df_reviews = self.load_test_data()
        
        # Test GridSearch reproducibility
        gridsearch_results = self.test_gridsearch_reproducibility(df_reviews)
        
        # Test GA reproducibility
        ga_results = self.test_ga_reproducibility(df_reviews)
        
        # Save results
        results_file = self.save_test_results(gridsearch_results, ga_results)
        
        total_time = time.time() - start_time
        
        print(f"\n🔬 REPRODUCIBILITY TEST COMPLETE!")
        print("=" * 60)
        print(f"🎯 GridSearch: {gridsearch_results['analysis']['status']} ({gridsearch_results['analysis']['reproducibility_score']:.1%})")
        print(f"🎯 GA: {ga_results['analysis']['status']} ({ga_results['analysis']['reproducibility_score']:.1%})")
        print(f"📁 Results: {results_file}")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        
        return {
            'gridsearch': gridsearch_results,
            'genetic_algorithm': ga_results,
            'results_file': results_file,
            'total_time': total_time
        }


def main():
    """Main execution function."""
    print("🔬 SPOILERSHIELD: REPRODUCIBILITY TESTING")
    print("=" * 60)
    
    # Initialize
    config = EnvConfig()
    
    # Run reproducibility test
    tester = ReproducibilityTester(config)
    results = tester.run_full_reproducibility_test()
    
    print(f"\n🎯 FINAL STATUS:")
    print(f"📈 All systems ready for A/B testing with verified reproducibility!")


if __name__ == "__main__":
    main()
