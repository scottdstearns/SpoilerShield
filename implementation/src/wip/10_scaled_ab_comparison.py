#!/usr/bin/env python3
"""
SpoilerShield: Scaled A/B Comparison - GridSearch vs Genetic Algorithm
======================================================================

Comprehensive A/B testing with:
- LogisticRegression: 288 combinations vs continuous optimization
- RoBERTa: 48 combinations vs continuous learning rate
- Multi-objective optimization (F1 + AUC + efficiency)
- Parallelized execution
- Comprehensive analysis and visualization

Author: SpoilerShield Development Team
Date: 2025-01-07
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
src_path = Path(__file__).parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader


def set_all_seeds(seed: int):
    """Set seeds for all random number generators for full reproducibility."""
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass
    
    plt.rcParams['figure.max_open_warning'] = 0
    print(f"🔒 All random seeds set to: {seed}")


class ScaledABAnalyzer:
    """
    Comprehensive A/B testing framework for scaled optimization comparison.
    """
    
    def __init__(self, config: EnvConfig, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        set_all_seeds(random_state)
        
        print("⚔️ SPOILERSHIELD: SCALED A/B COMPARISON")
        print("=" * 70)
    
    def run_scaled_gridsearch(self) -> Dict[str, Any]:
        """Run scaled GridSearch optimization."""
        print("\n📊 RUNNING SCALED GRIDSEARCH")
        print("=" * 50)
        
        result = subprocess.run([sys.executable, "08_scaled_gridsearch.py"], 
                              capture_output=True, text=True, cwd=src_path)
        
        if result.returncode != 0:
            print(f"❌ Scaled GridSearch failed: {result.stderr}")
            return None
        
        # Load results
        results_files = list(self.config.output_dir.glob("scaled_gridsearch_results_*.json"))
        if not results_files:
            print("❌ No GridSearch results found")
            return None
        
        latest_file = max(results_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def run_scaled_genetic_algorithm(self) -> Dict[str, Any]:
        """Run scaled Genetic Algorithm optimization."""
        print("\n🧬 RUNNING SCALED GENETIC ALGORITHM")
        print("=" * 50)
        
        result = subprocess.run([sys.executable, "09_scaled_genetic_algorithm.py"], 
                              capture_output=True, text=True, cwd=src_path)
        
        if result.returncode != 0:
            print(f"❌ Scaled GA failed: {result.stderr}")
            return None
        
        # Load results
        results_files = list(self.config.output_dir.glob("scaled_ga_results_*.json"))
        if not results_files:
            print("❌ No GA results found")
            return None
        
        latest_file = max(results_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def compare_results(self, gridsearch_results: Dict[str, Any], 
                       ga_results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive comparison of optimization methods."""
        print("\n⚔️ SCALED A/B COMPARISON ANALYSIS")
        print("=" * 60)
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'methods_compared': ['Scaled_GridSearch', 'Scaled_GeneticAlgorithm'],
            'logistic_regression': {},
            'roberta': {},
            'overall_analysis': {},
            'recommendations': {}
        }
        
        # LogisticRegression comparison
        print("📈 LOGISTIC REGRESSION COMPARISON")
        print("-" * 40)
        
        lr_comparison = self._compare_logistic_results(gridsearch_results, ga_results)
        comparison['logistic_regression'] = lr_comparison
        
        # RoBERTa comparison
        print("\n🤖 ROBERTA COMPARISON")
        print("-" * 40)
        
        roberta_comparison = self._compare_roberta_results(gridsearch_results, ga_results)
        comparison['roberta'] = roberta_comparison
        
        # Overall analysis
        print("\n🎯 OVERALL ANALYSIS")
        print("-" * 40)
        
        overall_analysis = self._analyze_overall_performance(lr_comparison, roberta_comparison)
        comparison['overall_analysis'] = overall_analysis
        
        # Recommendations
        comparison['recommendations'] = self._generate_recommendations(comparison)
        
        return comparison
    
    def _compare_logistic_results(self, grid_results: Dict[str, Any], ga_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare LogisticRegression optimization results."""
        grid_lr = grid_results.get('logistic_regression', {})
        ga_lr_strict = ga_results.get('logistic_strict', {})
        ga_lr_advantage = ga_results.get('logistic_advantage', {})
        
        comparison = {
            'performance': {},
            'efficiency': {},
            'parameter_discovery': {},
            'winner_analysis': {}
        }
        
        # Performance comparison
        grid_multi_obj = grid_lr.get('multi_objective_score', 0.0)
        ga_strict_multi_obj = ga_lr_strict.get('best_multi_objective', 0.0)
        ga_advantage_multi_obj = ga_lr_advantage.get('best_multi_objective', 0.0)
        
        grid_f1 = grid_lr.get('test_metrics', {}).get('f1', 0.0)
        grid_auc = grid_lr.get('test_metrics', {}).get('roc_auc', 0.0)
        
        ga_strict_f1 = ga_lr_strict.get('test_f1', 0.0)
        ga_strict_auc = ga_lr_strict.get('test_auc', 0.0)
        
        ga_advantage_f1 = ga_lr_advantage.get('test_f1', 0.0)
        ga_advantage_auc = ga_lr_advantage.get('test_auc', 0.0)
        
        comparison['performance'] = {
            'gridsearch': {
                'multi_objective': grid_multi_obj,
                'test_f1': grid_f1,
                'test_auc': grid_auc
            },
            'ga_strict': {
                'multi_objective': ga_strict_multi_obj,
                'test_f1': ga_strict_f1,
                'test_auc': ga_strict_auc
            },
            'ga_advantage': {
                'multi_objective': ga_advantage_multi_obj,
                'test_f1': ga_advantage_f1,
                'test_auc': ga_advantage_auc
            },
            'best_method': self._determine_best_method([
                ('GridSearch', grid_multi_obj),
                ('GA_Strict', ga_strict_multi_obj),
                ('GA_Advantage', ga_advantage_multi_obj)
            ])
        }
        
        # Efficiency comparison
        grid_time = grid_lr.get('optimization_time', 0.0)
        ga_strict_time = ga_lr_strict.get('optimization_time', 0.0)
        ga_advantage_time = ga_lr_advantage.get('optimization_time', 0.0)
        
        comparison['efficiency'] = {
            'gridsearch_time': grid_time,
            'ga_strict_time': ga_strict_time,
            'ga_advantage_time': ga_advantage_time,
            'fastest_method': self._determine_fastest_method([
                ('GridSearch', grid_time),
                ('GA_Strict', ga_strict_time),
                ('GA_Advantage', ga_advantage_time)
            ])
        }
        
        # Parameter discovery
        grid_params = grid_lr.get('best_params', {})
        ga_strict_params = ga_lr_strict.get('best_params', {})
        ga_advantage_params = ga_lr_advantage.get('best_params', {})
        
        comparison['parameter_discovery'] = {
            'gridsearch': grid_params,
            'ga_strict': ga_strict_params,
            'ga_advantage': ga_advantage_params,
            'continuous_advantage': self._analyze_continuous_advantage(grid_params, ga_advantage_params)
        }
        
        # Winner analysis
        comparison['winner_analysis'] = {
            'performance_winner': comparison['performance']['best_method'],
            'efficiency_winner': comparison['efficiency']['fastest_method'],
            'overall_recommendation': self._recommend_overall_method(comparison)
        }
        
        print(f"  Performance Winner: {comparison['winner_analysis']['performance_winner']}")
        print(f"  Efficiency Winner: {comparison['winner_analysis']['efficiency_winner']}")
        print(f"  Overall Recommendation: {comparison['winner_analysis']['overall_recommendation']}")
        
        return comparison
    
    def _compare_roberta_results(self, grid_results: Dict[str, Any], ga_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare RoBERTa optimization results."""
        grid_roberta = grid_results.get('roberta', {})
        ga_roberta_strict = ga_results.get('roberta_strict', {})
        ga_roberta_advantage = ga_results.get('roberta_advantage', {})
        
        # Check if transformers were available
        if 'error' in grid_roberta or 'error' in ga_roberta_strict or 'error' in ga_roberta_advantage:
            print("  ⚠️ RoBERTa comparison skipped - Transformers not available")
            return {'status': 'skipped', 'reason': 'Transformers not available'}
        
        comparison = {
            'performance': {},
            'efficiency': {},
            'parameter_discovery': {},
            'winner_analysis': {}
        }
        
        # Performance comparison
        grid_multi_obj = grid_roberta.get('multi_objective_score', 0.0)
        ga_strict_multi_obj = ga_roberta_strict.get('best_multi_objective', 0.0)
        ga_advantage_multi_obj = ga_roberta_advantage.get('best_multi_objective', 0.0)
        
        comparison['performance'] = {
            'gridsearch': {'multi_objective': grid_multi_obj},
            'ga_strict': {'multi_objective': ga_strict_multi_obj},
            'ga_advantage': {'multi_objective': ga_advantage_multi_obj},
            'best_method': self._determine_best_method([
                ('GridSearch', grid_multi_obj),
                ('GA_Strict', ga_strict_multi_obj),
                ('GA_Advantage', ga_advantage_multi_obj)
            ])
        }
        
        # Efficiency comparison
        grid_time = grid_roberta.get('optimization_time', 0.0)
        ga_strict_time = ga_roberta_strict.get('optimization_time', 0.0)
        ga_advantage_time = ga_roberta_advantage.get('optimization_time', 0.0)
        
        comparison['efficiency'] = {
            'gridsearch_time': grid_time,
            'ga_strict_time': ga_strict_time,
            'ga_advantage_time': ga_advantage_time,
            'fastest_method': self._determine_fastest_method([
                ('GridSearch', grid_time),
                ('GA_Strict', ga_strict_time),
                ('GA_Advantage', ga_advantage_time)
            ])
        }
        
        print(f"  Performance Winner: {comparison['performance']['best_method']}")
        print(f"  Efficiency Winner: {comparison['efficiency']['fastest_method']}")
        
        return comparison
    
    def _determine_best_method(self, method_scores: List[Tuple[str, float]]) -> str:
        """Determine best method by score."""
        return max(method_scores, key=lambda x: x[1])[0]
    
    def _determine_fastest_method(self, method_times: List[Tuple[str, float]]) -> str:
        """Determine fastest method by time."""
        return min(method_times, key=lambda x: x[1])[0]
    
    def _analyze_continuous_advantage(self, grid_params: Dict[str, Any], ga_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze advantage of continuous parameter optimization."""
        analysis = {}
        
        # Check if GA found continuous C value not in GridSearch discrete set
        grid_c = grid_params.get('classifier__C', 0.0)
        ga_c = ga_params.get('classifier__C', 0.0)
        
        # GridSearch discrete C values
        discrete_c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        # Check if GA found value between discrete points
        is_between_discrete = any(
            discrete_c_values[i] < ga_c < discrete_c_values[i+1] 
            for i in range(len(discrete_c_values)-1)
        )
        
        analysis['continuous_c_advantage'] = {
            'grid_c': grid_c,
            'ga_c': ga_c,
            'is_between_discrete_values': is_between_discrete,
            'exploration_benefit': is_between_discrete
        }
        
        return analysis
    
    def _recommend_overall_method(self, comparison: Dict[str, Any]) -> str:
        """Recommend overall best method based on multiple criteria."""
        perf_winner = comparison['winner_analysis']['performance_winner']
        eff_winner = comparison['winner_analysis']['efficiency_winner']
        
        # Simple heuristic: prefer performance unless efficiency gap is huge
        perf_scores = comparison['performance']
        eff_times = comparison['efficiency']
        
        # If GA advantage wins performance and time is reasonable, recommend it
        if perf_winner == 'GA_Advantage':
            return 'GA_Advantage (Performance + Continuous Optimization)'
        elif perf_winner == 'GridSearch' and eff_winner == 'GridSearch':
            return 'GridSearch (Performance + Efficiency)'
        else:
            return f'{perf_winner} (Performance Winner)'
    
    def _analyze_overall_performance(self, lr_comparison: Dict[str, Any], 
                                   roberta_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall performance across all models."""
        analysis = {
            'logistic_regression_winner': lr_comparison.get('winner_analysis', {}).get('overall_recommendation', 'Unknown'),
            'roberta_winner': roberta_comparison.get('winner_analysis', {}).get('overall_recommendation', 'Unknown') if 'status' not in roberta_comparison else 'Skipped',
            'ga_advantage_demonstrated': False,
            'continuous_optimization_benefit': False
        }
        
        # Check if GA advantage shows benefit
        lr_perf = lr_comparison.get('performance', {})
        if lr_perf:
            ga_adv_score = lr_perf.get('ga_advantage', {}).get('multi_objective', 0.0)
            grid_score = lr_perf.get('gridsearch', {}).get('multi_objective', 0.0)
            ga_strict_score = lr_perf.get('ga_strict', {}).get('multi_objective', 0.0)
            
            analysis['ga_advantage_demonstrated'] = ga_adv_score > max(grid_score, ga_strict_score)
        
        # Check continuous optimization benefit
        param_discovery = lr_comparison.get('parameter_discovery', {})
        if param_discovery:
            cont_adv = param_discovery.get('continuous_advantage', {})
            analysis['continuous_optimization_benefit'] = cont_adv.get('exploration_benefit', False)
        
        return analysis
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> Dict[str, str]:
        """Generate actionable recommendations."""
        recommendations = {}
        
        overall = comparison.get('overall_analysis', {})
        
        if overall.get('ga_advantage_demonstrated', False):
            recommendations['primary'] = 'Use Genetic Algorithm with continuous parameter optimization'
            recommendations['reason'] = 'GA advantage mode shows superior multi-objective performance'
        else:
            recommendations['primary'] = 'Consider GridSearch for reliability, GA for exploration'
            recommendations['reason'] = 'Mixed results suggest method choice depends on specific requirements'
        
        if overall.get('continuous_optimization_benefit', False):
            recommendations['continuous_params'] = 'Continuous parameter optimization provides measurable benefit'
        else:
            recommendations['continuous_params'] = 'Discrete parameter space may be sufficient for this problem'
        
        recommendations['scaling'] = 'Framework is ready for larger parameter spaces where GA advantages will be more pronounced'
        recommendations['production'] = 'Use best-performing method from this analysis for final model deployment'
        
        return recommendations
    
    def create_comprehensive_visualization(self, comparison: Dict[str, Any]) -> str:
        """Create comprehensive visualization of A/B test results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Scaled A/B Test: GridSearch vs Genetic Algorithm', fontsize=16, fontweight='bold')
        
        # LogisticRegression performance comparison
        lr_perf = comparison.get('logistic_regression', {}).get('performance', {})
        if lr_perf:
            methods = ['GridSearch', 'GA Strict', 'GA Advantage']
            multi_obj_scores = [
                lr_perf.get('gridsearch', {}).get('multi_objective', 0.0),
                lr_perf.get('ga_strict', {}).get('multi_objective', 0.0),
                lr_perf.get('ga_advantage', {}).get('multi_objective', 0.0)
            ]
            
            axes[0, 0].bar(methods, multi_obj_scores, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
            axes[0, 0].set_title('LogReg: Multi-Objective Score')
            axes[0, 0].set_ylabel('Multi-Objective Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(multi_obj_scores):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # LogisticRegression efficiency comparison
        lr_eff = comparison.get('logistic_regression', {}).get('efficiency', {})
        if lr_eff:
            times = [
                lr_eff.get('gridsearch_time', 0.0),
                lr_eff.get('ga_strict_time', 0.0),
                lr_eff.get('ga_advantage_time', 0.0)
            ]
            
            axes[0, 1].bar(methods, times, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
            axes[0, 1].set_title('LogReg: Optimization Time')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(times):
                axes[0, 1].text(i, v + max(times)*0.02, f'{v:.1f}s', ha='center', va='bottom')
        
        # Parameter comparison (C values)
        lr_params = comparison.get('logistic_regression', {}).get('parameter_discovery', {})
        if lr_params:
            grid_c = lr_params.get('gridsearch', {}).get('classifier__C', 0.0)
            ga_strict_c = lr_params.get('ga_strict', {}).get('classifier__C', 0.0)
            ga_advantage_c = lr_params.get('ga_advantage', {}).get('classifier__C', 0.0)
            
            c_values = [grid_c, ga_strict_c, ga_advantage_c]
            
            axes[0, 2].bar(methods, c_values, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
            axes[0, 2].set_title('LogReg: Optimal C Parameter')
            axes[0, 2].set_ylabel('C Value')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(c_values):
                axes[0, 2].text(i, v + max(c_values)*0.02, f'{v:.2f}', ha='center', va='bottom')
        
        # RoBERTa comparison (if available)
        roberta_comp = comparison.get('roberta', {})
        if 'status' not in roberta_comp:
            roberta_perf = roberta_comp.get('performance', {})
            roberta_eff = roberta_comp.get('efficiency', {})
            
            if roberta_perf:
                roberta_scores = [
                    roberta_perf.get('gridsearch', {}).get('multi_objective', 0.0),
                    roberta_perf.get('ga_strict', {}).get('multi_objective', 0.0),
                    roberta_perf.get('ga_advantage', {}).get('multi_objective', 0.0)
                ]
                
                axes[1, 0].bar(methods, roberta_scores, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
                axes[1, 0].set_title('RoBERTa: Multi-Objective Score')
                axes[1, 0].set_ylabel('Multi-Objective Score')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            if roberta_eff:
                roberta_times = [
                    roberta_eff.get('gridsearch_time', 0.0),
                    roberta_eff.get('ga_strict_time', 0.0),
                    roberta_eff.get('ga_advantage_time', 0.0)
                ]
                
                axes[1, 1].bar(methods, roberta_times, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
                axes[1, 1].set_title('RoBERTa: Optimization Time')
                axes[1, 1].set_ylabel('Time (seconds)')
                axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            # RoBERTa not available
            axes[1, 0].text(0.5, 0.5, 'RoBERTa\nNot Available', ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 1].text(0.5, 0.5, 'RoBERTa\nNot Available', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        
        # Overall summary
        overall = comparison.get('overall_analysis', {})
        summary_text = f"""SCALED A/B TEST SUMMARY

LogReg Winner: {overall.get('logistic_regression_winner', 'Unknown')}

RoBERTa Winner: {overall.get('roberta_winner', 'Unknown')}

GA Advantage: {'✅ Demonstrated' if overall.get('ga_advantage_demonstrated', False) else '❌ Not Clear'}

Continuous Benefit: {'✅ Yes' if overall.get('continuous_optimization_benefit', False) else '❌ Minimal'}

Recommendation: {comparison.get('recommendations', {}).get('primary', 'See detailed analysis')}
"""
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.config.output_dir / f"scaled_ab_comparison_visualization_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def save_comparison_report(self, comparison: Dict[str, Any]) -> str:
        """Save comprehensive A/B comparison report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.config.output_dir / f"scaled_ab_comparison_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Create markdown report
        md_file = self.config.output_dir / f"scaled_ab_comparison_report_{timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write("# Scaled A/B Comparison: GridSearch vs Genetic Algorithm\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            recommendations = comparison.get('recommendations', {})
            f.write(f"**Primary Recommendation:** {recommendations.get('primary', 'See analysis')}\n\n")
            f.write(f"**Reason:** {recommendations.get('reason', 'Mixed results')}\n\n")
            
            # LogisticRegression Results
            f.write("## 📊 LogisticRegression Results\n\n")
            lr_comp = comparison.get('logistic_regression', {})
            
            if 'performance' in lr_comp:
                perf = lr_comp['performance']
                f.write("### Performance Comparison\n\n")
                f.write("| Method | Multi-Objective Score | Test F1 | Test AUC |\n")
                f.write("|--------|----------------------|---------|----------|\n")
                
                grid_perf = perf.get('gridsearch', {})
                ga_strict_perf = perf.get('ga_strict', {})
                ga_adv_perf = perf.get('ga_advantage', {})
                
                f.write(f"| GridSearch | {grid_perf.get('multi_objective', 0.0):.4f} | {grid_perf.get('test_f1', 0.0):.4f} | {grid_perf.get('test_auc', 0.0):.4f} |\n")
                f.write(f"| GA Strict | {ga_strict_perf.get('multi_objective', 0.0):.4f} | {ga_strict_perf.get('test_f1', 0.0):.4f} | {ga_strict_perf.get('test_auc', 0.0):.4f} |\n")
                f.write(f"| GA Advantage | {ga_adv_perf.get('multi_objective', 0.0):.4f} | {ga_adv_perf.get('test_f1', 0.0):.4f} | {ga_adv_perf.get('test_auc', 0.0):.4f} |\n\n")
                
                f.write(f"**Winner:** {perf.get('best_method', 'Unknown')}\n\n")
            
            if 'efficiency' in lr_comp:
                eff = lr_comp['efficiency']
                f.write("### Efficiency Comparison\n\n")
                f.write("| Method | Optimization Time |\n")
                f.write("|--------|-----------------|\n")
                f.write(f"| GridSearch | {eff.get('gridsearch_time', 0.0):.1f}s |\n")
                f.write(f"| GA Strict | {eff.get('ga_strict_time', 0.0):.1f}s |\n")
                f.write(f"| GA Advantage | {eff.get('ga_advantage_time', 0.0):.1f}s |\n\n")
                
                f.write(f"**Fastest:** {eff.get('fastest_method', 'Unknown')}\n\n")
            
            # RoBERTa Results (if available)
            roberta_comp = comparison.get('roberta', {})
            if 'status' not in roberta_comp:
                f.write("## 🤖 RoBERTa Results\n\n")
                f.write("### Performance Comparison\n\n")
                # Add RoBERTa results here
            else:
                f.write("## 🤖 RoBERTa Results\n\n")
                f.write("⚠️ RoBERTa comparison was skipped (Transformers not available)\n\n")
            
            # Key Findings
            f.write("## 🔍 Key Findings\n\n")
            overall = comparison.get('overall_analysis', {})
            
            f.write(f"- **GA Advantage Demonstrated:** {'✅ Yes' if overall.get('ga_advantage_demonstrated', False) else '❌ No'}\n")
            f.write(f"- **Continuous Optimization Benefit:** {'✅ Yes' if overall.get('continuous_optimization_benefit', False) else '❌ Minimal'}\n")
            f.write(f"- **Scaling Recommendation:** {recommendations.get('scaling', 'Continue testing')}\n")
            f.write(f"- **Production Recommendation:** {recommendations.get('production', 'Use best method')}\n\n")
            
            # Technical Details
            f.write("## 🔧 Technical Details\n\n")
            f.write("### GridSearch Configuration\n")
            f.write("- **LogReg Parameter Space:** 288 combinations (4×6×3×4)\n")
            f.write("- **RoBERTa Parameter Space:** 48 combinations (2×6×2×2)\n")
            f.write("- **Cross-Validation:** 3-fold stratified\n")
            f.write("- **Parallelization:** n_jobs=-1\n\n")
            
            f.write("### Genetic Algorithm Configuration\n")
            f.write("- **Population Size:** 40 individuals\n")
            f.write("- **Generations:** 15\n")
            f.write("- **Parallelization:** joblib with all CPU cores\n")
            f.write("- **Selection:** Tournament (size=5)\n")
            f.write("- **Crossover:** Uniform (rate=0.8)\n")
            f.write("- **Mutation:** Adaptive (initial=0.2, decay over generations)\n\n")
            
            f.write("### Multi-Objective Scoring\n")
            f.write("- **F1 Weight:** 0.4\n")
            f.write("- **AUC Weight:** 0.5 (stronger emphasis)\n")
            f.write("- **Efficiency Weight:** 0.1 (weaker emphasis)\n")
        
        print(f"📋 Comprehensive report saved: {md_file}")
        return str(md_file)
    
    def run_scaled_ab_comparison(self) -> Dict[str, Any]:
        """Run complete scaled A/B comparison."""
        print("⚔️ STARTING SCALED A/B COMPARISON")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run GridSearch
        gridsearch_results = self.run_scaled_gridsearch()
        if not gridsearch_results:
            return {'error': 'GridSearch failed'}
        
        # Run Genetic Algorithm
        ga_results = self.run_scaled_genetic_algorithm()
        if not ga_results:
            return {'error': 'Genetic Algorithm failed'}
        
        # Compare results
        comparison = self.compare_results(gridsearch_results, ga_results)
        
        # Create visualization
        plot_file = self.create_comprehensive_visualization(comparison)
        
        # Save comprehensive report
        report_file = self.save_comparison_report(comparison)
        
        total_time = time.time() - start_time
        
        print(f"\n⚔️ SCALED A/B COMPARISON COMPLETE!")
        print("=" * 70)
        
        overall = comparison.get('overall_analysis', {})
        print(f"🏆 GA Advantage Demonstrated: {'✅ Yes' if overall.get('ga_advantage_demonstrated', False) else '❌ No'}")
        print(f"📊 Continuous Optimization Benefit: {'✅ Yes' if overall.get('continuous_optimization_benefit', False) else '❌ Minimal'}")
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
    print("⚔️ SPOILERSHIELD: SCALED A/B COMPARISON")
    print("=" * 70)
    
    # Initialize
    config = EnvConfig()
    
    # Run scaled A/B comparison
    analyzer = ScaledABAnalyzer(config)
    results = analyzer.run_scaled_ab_comparison()
    
    if 'error' in results:
        print(f"❌ A/B comparison failed: {results['error']}")
        return
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"📈 Comprehensive analysis completed successfully")
    print(f"📋 Full report and visualizations available in outputs directory")


if __name__ == "__main__":
    main()
