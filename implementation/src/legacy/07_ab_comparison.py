#!/usr/bin/env python3
"""
SpoilerShield: A/B Comparison - GridSearch vs Genetic Algorithm
===============================================================

This script runs both optimization methods and provides comprehensive A/B analysis
comparing GridSearchCV vs Genetic Algorithm approaches for hyperparameter optimization.

Author: SpoilerShield Development Team
Date: 2025-08-07
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import numpy as np

# Add src to path
src_path = Path(__file__).parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader
import importlib.util

# Dynamic imports for the optimization modules
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
    
    # Try to set PyTorch seeds (if available)
    try:
        import torch
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
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass  # PyTorch/transformers not available
    
    # Matplotlib for consistent plot generation
    plt.rcParams['figure.max_open_warning'] = 0
    
    print(f"🔒 All random seeds set to: {seed}")


class ABComparisonAnalyzer:
    """
    A/B testing framework for comparing GridSearch vs Genetic Algorithm optimization.
    """
    
    def __init__(self, config: EnvConfig, random_state: int = 42):
        """Initialize the A/B comparison analyzer."""
        self.config = config
        self.random_state = random_state
        self.results = {}
        
        # Set all random seeds for reproducibility
        set_all_seeds(random_state)
        
        print("⚔️ SPOILERSHIELD: A/B COMPARISON - GRIDSEARCH VS GENETIC ALGORITHM")
        print("=" * 70)
    
    def run_gridsearch_baseline(self, df_reviews: pd.DataFrame) -> Dict[str, Any]:
        """Run simplified GridSearch optimization."""
        print("\n📊 RUNNING GRIDSEARCH BASELINE")
        print("=" * 50)
        
        optimizer = SimplifiedGridSearchOptimizer(self.config)
        return optimizer.run_simplified_optimization(df_reviews)
    
    def run_genetic_algorithm(self, df_reviews: pd.DataFrame) -> Dict[str, Any]:
        """Run Genetic Algorithm optimization."""
        print("\n🧬 RUNNING GENETIC ALGORITHM")
        print("=" * 50)
        
        optimizer = GeneticHyperparameterOptimizer(self.config)
        return optimizer.run_genetic_optimization(df_reviews)
    
    def compare_results(self, gridsearch_results: Dict[str, Any], 
                       ga_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive comparison of optimization methods.
        """
        print("\n⚔️ A/B COMPARISON ANALYSIS")
        print("=" * 50)
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'methods_compared': ['GridSearchCV_Simplified', 'GeneticAlgorithm'],
            'performance_comparison': {},
            'efficiency_comparison': {},
            'parameter_exploration': {},
            'winner_analysis': {},
            'recommendations': {}
        }
        
        # Performance Comparison
        print("📈 PERFORMANCE COMPARISON")
        print("-" * 30)
        
        perf_comparison = {}
        for model in ['logistic_regression', 'roberta']:
            if (model in gridsearch_results and 'error' not in gridsearch_results[model] and
                model in ga_results and 'error' not in ga_results[model]):
                
                # Extract F1 scores
                gs_f1 = gridsearch_results[model].get('test_metrics', {}).get('f1', 0)
                if gs_f1 == 0:  # Try alternative key structure
                    gs_f1 = gridsearch_results[model].get('test_f1', 0)
                
                ga_f1 = ga_results[model].get('test_f1', 0)
                
                # Performance difference
                perf_diff = ga_f1 - gs_f1
                perf_improvement = (perf_diff / gs_f1 * 100) if gs_f1 > 0 else 0
                
                perf_comparison[model] = {
                    'gridsearch_f1': gs_f1,
                    'genetic_f1': ga_f1,
                    'difference': perf_diff,
                    'improvement_percent': perf_improvement,
                    'winner': 'Genetic Algorithm' if ga_f1 > gs_f1 else 'GridSearch'
                }
                
                print(f"  {model.replace('_', ' ').title()}:")
                print(f"    GridSearch F1: {gs_f1:.4f}")
                print(f"    Genetic F1: {ga_f1:.4f}")
                print(f"    Difference: {perf_diff:+.4f} ({perf_improvement:+.1f}%)")
                print(f"    Winner: {perf_comparison[model]['winner']}")
        
        comparison['performance_comparison'] = perf_comparison
        
        # Efficiency Comparison
        print(f"\n⚡ EFFICIENCY COMPARISON")
        print("-" * 30)
        
        efficiency_comparison = {}
        for model in ['logistic_regression', 'roberta']:
            if (model in gridsearch_results and 'error' not in gridsearch_results[model] and
                model in ga_results and 'error' not in ga_results[model]):
                
                # Extract optimization times
                gs_time = gridsearch_results[model].get('optimization_time', 0)
                ga_time = ga_results[model].get('optimization_time', 0)
                
                # Time difference
                time_diff = ga_time - gs_time
                time_ratio = ga_time / gs_time if gs_time > 0 else float('inf')
                
                # Evaluations comparison
                gs_evals = gridsearch_results[model].get('total_combinations', 0) * 3  # 3-fold CV
                ga_evals = ga_results[model].get('total_evaluations', 0)
                
                efficiency_comparison[model] = {
                    'gridsearch_time': gs_time,
                    'genetic_time': ga_time,
                    'time_difference': time_diff,
                    'time_ratio': time_ratio,
                    'gridsearch_evaluations': gs_evals,
                    'genetic_evaluations': ga_evals,
                    'efficiency_winner': 'GridSearch' if gs_time < ga_time else 'Genetic Algorithm'
                }
                
                print(f"  {model.replace('_', ' ').title()}:")
                print(f"    GridSearch time: {gs_time:.1f}s ({gs_evals} evaluations)")
                print(f"    Genetic time: {ga_time:.1f}s ({ga_evals} evaluations)")
                print(f"    Time ratio (GA/GS): {time_ratio:.2f}x")
                print(f"    Efficiency winner: {efficiency_comparison[model]['efficiency_winner']}")
        
        comparison['efficiency_comparison'] = efficiency_comparison
        
        # Parameter Exploration Analysis
        print(f"\n🔍 PARAMETER EXPLORATION ANALYSIS")
        print("-" * 30)
        
        exploration_analysis = {}
        for model in ['logistic_regression', 'roberta']:
            if model in ga_results and 'error' not in ga_results[model]:
                gs_params = gridsearch_results.get(model, {}).get('best_params', {})
                ga_params = ga_results[model].get('best_params', {})
                
                exploration_analysis[model] = {
                    'gridsearch_params': gs_params,
                    'genetic_params': ga_params,
                    'parameter_similarity': self._calculate_parameter_similarity(gs_params, ga_params)
                }
                
                print(f"  {model.replace('_', ' ').title()}:")
                print(f"    GridSearch best: {gs_params}")
                print(f"    Genetic best: {ga_params}")
                print(f"    Similarity: {exploration_analysis[model]['parameter_similarity']:.2f}")
        
        comparison['parameter_exploration'] = exploration_analysis
        
        # Overall Winner Analysis
        print(f"\n🏆 OVERALL WINNER ANALYSIS")
        print("-" * 30)
        
        winner_analysis = self._determine_overall_winner(perf_comparison, efficiency_comparison)
        comparison['winner_analysis'] = winner_analysis
        
        for criterion, winner in winner_analysis.items():
            print(f"  {criterion}: {winner}")
        
        # Recommendations
        comparison['recommendations'] = self._generate_recommendations(comparison)
        
        return comparison
    
    def _calculate_parameter_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate similarity between two parameter sets."""
        if not params1 or not params2:
            return 0.0
        
        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return 0.0
        
        similarity_score = 0.0
        for param in common_params:
            val1, val2 = params1[param], params2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity (normalized by relative difference)
                if val1 == val2:
                    similarity_score += 1.0
                else:
                    relative_diff = abs(val1 - val2) / max(abs(val1), abs(val2), 1e-8)
                    similarity_score += max(0, 1.0 - relative_diff)
            else:
                # Categorical similarity
                similarity_score += 1.0 if val1 == val2 else 0.0
        
        return similarity_score / len(common_params)
    
    def _determine_overall_winner(self, perf_comparison: Dict[str, Any], 
                                 efficiency_comparison: Dict[str, Any]) -> Dict[str, str]:
        """Determine overall winners across different criteria."""
        winner_analysis = {}
        
        # Performance winner
        perf_wins = sum(1 for model_data in perf_comparison.values() 
                       if model_data['winner'] == 'Genetic Algorithm')
        total_models = len(perf_comparison)
        
        if perf_wins > total_models / 2:
            winner_analysis['Performance'] = 'Genetic Algorithm'
        elif perf_wins < total_models / 2:
            winner_analysis['Performance'] = 'GridSearch'
        else:
            winner_analysis['Performance'] = 'Tie'
        
        # Efficiency winner
        eff_wins = sum(1 for model_data in efficiency_comparison.values() 
                      if model_data['efficiency_winner'] == 'Genetic Algorithm')
        
        if eff_wins > total_models / 2:
            winner_analysis['Efficiency'] = 'Genetic Algorithm'
        elif eff_wins < total_models / 2:
            winner_analysis['Efficiency'] = 'GridSearch'
        else:
            winner_analysis['Efficiency'] = 'Tie'
        
        # Overall recommendation
        if winner_analysis['Performance'] == 'Genetic Algorithm':
            if winner_analysis['Efficiency'] == 'Genetic Algorithm':
                winner_analysis['Overall'] = 'Genetic Algorithm (Performance + Efficiency)'
            else:
                winner_analysis['Overall'] = 'Genetic Algorithm (Performance > Efficiency)'
        elif winner_analysis['Performance'] == 'GridSearch':
            if winner_analysis['Efficiency'] == 'GridSearch':
                winner_analysis['Overall'] = 'GridSearch (Performance + Efficiency)'
            else:
                winner_analysis['Overall'] = 'GridSearch (Performance > Efficiency)'
        else:
            # Tie in performance, decide by efficiency
            winner_analysis['Overall'] = f"{winner_analysis['Efficiency']} (Efficiency tiebreaker)"
        
        return winner_analysis
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> Dict[str, str]:
        """Generate actionable recommendations based on comparison."""
        recommendations = {}
        
        winner = comparison['winner_analysis']['Overall']
        
        if 'Genetic Algorithm' in winner:
            recommendations['Primary'] = "Use Genetic Algorithm for hyperparameter optimization"
            recommendations['Reason'] = "GA shows better performance and/or efficiency"
            recommendations['When_to_use_GA'] = "Complex parameter spaces, multi-objective optimization, limited time budget"
            recommendations['When_to_use_GridSearch'] = "Simple parameter spaces, need guaranteed exhaustive search"
        else:
            recommendations['Primary'] = "Use GridSearch for hyperparameter optimization"
            recommendations['Reason'] = "GridSearch shows better performance and/or efficiency"
            recommendations['When_to_use_GridSearch'] = "Guaranteed optimal within search space, simple parameter spaces"
            recommendations['When_to_use_GA'] = "Very large parameter spaces, multi-objective optimization"
        
        recommendations['Hybrid_Approach'] = "Consider GA for exploration + local GridSearch for exploitation"
        recommendations['Production'] = "Use best-performing method from this A/B test for production deployment"
        
        return recommendations
    
    def create_visualization(self, comparison: Dict[str, Any]) -> str:
        """Create visualization comparing the methods."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance comparison
        models = list(comparison['performance_comparison'].keys())
        gs_f1s = [comparison['performance_comparison'][m]['gridsearch_f1'] for m in models]
        ga_f1s = [comparison['performance_comparison'][m]['genetic_f1'] for m in models]
        
        x = range(len(models))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], gs_f1s, width, label='GridSearch', alpha=0.8, color='skyblue')
        ax1.bar([i + width/2 for i in x], ga_f1s, width, label='Genetic Algorithm', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Performance Comparison (F1 Score)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', '\n') for m in models])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency comparison
        gs_times = [comparison['efficiency_comparison'][m]['gridsearch_time'] for m in models]
        ga_times = [comparison['efficiency_comparison'][m]['genetic_time'] for m in models]
        
        ax2.bar([i - width/2 for i in x], gs_times, width, label='GridSearch', alpha=0.8, color='skyblue')
        ax2.bar([i + width/2 for i in x], ga_times, width, label='Genetic Algorithm', alpha=0.8, color='lightcoral')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Optimization Time (seconds)')
        ax2.set_title('Efficiency Comparison (Time)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', '\n') for m in models])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance improvement
        improvements = [comparison['performance_comparison'][m]['improvement_percent'] for m in models]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        ax3.bar(range(len(models)), improvements, color=colors, alpha=0.7)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('GA Performance Improvement over GridSearch')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.replace('_', '\n') for m in models])
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # Time ratio
        time_ratios = [comparison['efficiency_comparison'][m]['time_ratio'] for m in models]
        colors = ['green' if ratio < 1 else 'red' for ratio in time_ratios]
        
        ax4.bar(range(len(models)), time_ratios, color=colors, alpha=0.7)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Time Ratio (GA/GridSearch)')
        ax4.set_title('Time Efficiency Ratio')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.replace('_', '\n') for m in models])
        ax4.axhline(y=1, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.config.output_dir / f"ab_comparison_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def save_comparison_report(self, comparison: Dict[str, Any]) -> str:
        """Save comprehensive A/B comparison report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.config.output_dir / f"ab_comparison_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Create markdown report
        md_file = self.config.output_dir / f"ab_comparison_report_{timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write("# SpoilerShield: A/B Comparison - GridSearch vs Genetic Algorithm\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            f.write(f"**Overall Winner:** {comparison['winner_analysis']['Overall']}\n\n")
            
            # Performance Results
            f.write("## 📈 Performance Comparison\n\n")
            f.write("| Model | GridSearch F1 | Genetic F1 | Improvement | Winner |\n")
            f.write("|-------|---------------|------------|-------------|--------|\n")
            
            for model, data in comparison['performance_comparison'].items():
                f.write(f"| {model.replace('_', ' ').title()} | {data['gridsearch_f1']:.4f} | "
                       f"{data['genetic_f1']:.4f} | {data['improvement_percent']:+.1f}% | {data['winner']} |\n")
            f.write("\n")
            
            # Efficiency Results
            f.write("## ⚡ Efficiency Comparison\n\n")
            f.write("| Model | GridSearch Time | Genetic Time | Time Ratio | Winner |\n")
            f.write("|-------|-----------------|--------------|------------|--------|\n")
            
            for model, data in comparison['efficiency_comparison'].items():
                f.write(f"| {model.replace('_', ' ').title()} | {data['gridsearch_time']:.1f}s | "
                       f"{data['genetic_time']:.1f}s | {data['time_ratio']:.2f}x | {data['efficiency_winner']} |\n")
            f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            for key, value in comparison['recommendations'].items():
                f.write(f"**{key.replace('_', ' ')}:** {value}\n\n")
            
            # Technical Details
            f.write("## 🔧 Technical Details\n\n")
            f.write("### Parameter Space Explored\n\n")
            f.write("**LogisticRegression:**\n")
            f.write("- `max_features`: [10000, 20000]\n")
            f.write("- `C`: [1.0, 5.0]\n")
            f.write("- `penalty`: ['l2']\n\n")
            
            f.write("**RoBERTa:**\n")
            f.write("- `learning_rate`: [3e-5, 5e-5]\n")
            f.write("- `max_length`: [256, 512]\n\n")
            
            # Methodology
            f.write("## 📋 Methodology\n\n")
            f.write("- **GridSearch**: Exhaustive search with 3-fold stratified cross-validation\n")
            f.write("- **Genetic Algorithm**: Population=20, Generations=10, Multi-objective fitness\n")
            f.write("- **Evaluation**: F1 score on held-out test set\n")
            f.write("- **Data Split**: 80% train, 20% test with stratification\n\n")
        
        print(f"📋 A/B comparison report saved: {md_file}")
        return str(md_file)
    
    def run_ab_comparison(self, df_reviews: pd.DataFrame) -> Dict[str, Any]:
        """Run complete A/B comparison."""
        print("⚔️ STARTING A/B COMPARISON")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run GridSearch
        gridsearch_results = self.run_gridsearch_baseline(df_reviews)
        
        # Run Genetic Algorithm
        ga_results = self.run_genetic_algorithm(df_reviews)
        
        # Compare results
        comparison = self.compare_results(gridsearch_results, ga_results)
        
        # Create visualization
        plot_file = self.create_visualization(comparison)
        
        # Save comprehensive report
        report_file = self.save_comparison_report(comparison)
        
        total_time = time.time() - start_time
        
        print(f"\n⚔️ A/B COMPARISON COMPLETE!")
        print("=" * 70)
        print(f"🏆 Overall Winner: {comparison['winner_analysis']['Overall']}")
        print(f"📁 Report: {report_file}")
        print(f"📊 Visualization: {plot_file}")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        
        return {
            'comparison': comparison,
            'gridsearch_results': gridsearch_results,
            'ga_results': ga_results,
            'report_file': report_file,
            'plot_file': plot_file,
            'total_time': total_time
        }


def main():
    """Main execution function."""
    print("⚔️ SPOILERSHIELD: A/B COMPARISON - GRIDSEARCH VS GENETIC ALGORITHM")
    print("=" * 70)
    
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
    
    # Run A/B comparison with fixed seed for reproducibility
    analyzer = ABComparisonAnalyzer(config, random_state=42)
    results = analyzer.run_ab_comparison(df_reviews)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"📈 Winner: {results['comparison']['winner_analysis']['Overall']}")
    print(f"📋 Full report available in outputs directory")


if __name__ == "__main__":
    main()
