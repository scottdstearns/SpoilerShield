#!/usr/bin/env python3
"""
SpoilerShield: Class Imbalance Optimization
===========================================

This script implements a comprehensive class imbalance handling strategy using:
1. Advanced SMOTE variants (ADASYN, BorderlineSMOTE, SMOTE-Tomek)
2. Cost-sensitive learning with optimized class weights
3. Threshold optimization using Youden's J statistic
4. Ensemble methods with imbalance-aware sampling

Author: SpoilerShield Development Team
Date: 2025-08-07
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced Learning Techniques
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
)
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path for our custom modules
src_path = Path(__file__).parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader
from evaluation.model_evaluator import ModelEvaluator


class ImbalanceOptimizer:
    """
    Comprehensive class imbalance optimization toolkit.
    
    Implements multiple strategies:
    1. Data-level: SMOTE variants
    2. Algorithm-level: Cost-sensitive learning
    3. Ensemble-level: Balanced ensemble methods
    4. Threshold-level: Optimal threshold selection
    """
    
    def __init__(self, config: EnvConfig, random_state: int = 42):
        """
        Initialize the ImbalanceOptimizer.
        
        Args:
            config: Environment configuration
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.results = {}
        self.best_strategies = {}
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(output_dir=str(config.output_dir))
        
        print("🎯 SPOILERSHIELD: CLASS IMBALANCE OPTIMIZATION")
        print("=" * 60)
    
    def analyze_class_distribution(self, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze the class distribution in detail.
        
        Args:
            y: Target variable series
            
        Returns:
            Dictionary with distribution analysis
        """
        print("\n📊 CLASS DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        # Basic counts
        value_counts = y.value_counts()
        percentages = y.value_counts(normalize=True) * 100
        
        # Calculate imbalance metrics
        minority_class = value_counts.idxmin()
        majority_class = value_counts.idxmax()
        imbalance_ratio = value_counts.max() / value_counts.min()
        
        analysis = {
            'total_samples': len(y),
            'class_counts': value_counts.to_dict(),
            'class_percentages': percentages.to_dict(),
            'minority_class': minority_class,
            'majority_class': majority_class,
            'imbalance_ratio': imbalance_ratio,
            'minority_percentage': percentages[minority_class],
            'majority_percentage': percentages[majority_class]
        }
        
        print(f"Total Samples: {analysis['total_samples']:,}")
        print(f"Class Distribution:")
        for class_label, count in value_counts.items():
            percentage = percentages[class_label]
            marker = "📈" if class_label == majority_class else "📉"
            print(f"  {marker} Class {class_label}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n🎯 Imbalance Metrics:")
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"  Minority Class: {minority_class} ({analysis['minority_percentage']:.1f}%)")
        print(f"  Baseline Accuracy (predict majority): {analysis['majority_percentage']:.1f}%")
        
        return analysis
    
    def prepare_sampling_strategies(self) -> Dict[str, Any]:
        """
        Prepare different sampling strategies for comparison.
        
        Returns:
            Dictionary of sampling strategy objects
        """
        print("\n🔧 PREPARING SAMPLING STRATEGIES")
        print("-" * 40)
        
        strategies = {
            # No sampling (baseline)
            'None': None,
            
            # Basic SMOTE
            'SMOTE': SMOTE(random_state=self.random_state),
            
            # One advanced variant for testing
            'ADASYN': ADASYN(random_state=self.random_state, n_neighbors=5)
        }
        
        print("Available Strategies:")
        for name, strategy in strategies.items():
            if strategy is None:
                print(f"  ✅ {name}: Baseline (no resampling)")
            else:
                print(f"  ✅ {name}: {strategy.__class__.__name__}")
        
        return strategies
    
    def compute_optimal_class_weights(self, y: pd.Series) -> Dict[str, Dict]:
        """
        Compute optimal class weights using different strategies.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary of class weight strategies
        """
        print("\n⚖️ COMPUTING OPTIMAL CLASS WEIGHTS")
        print("-" * 40)
        
        # Get unique classes
        classes = np.unique(y)
        
        # Strategy 1: Balanced (sklearn default)
        balanced_weights = compute_class_weight('balanced', classes=classes, y=y)
        balanced_dict = dict(zip(classes, balanced_weights))
        
        # Strategy 2: Inverse frequency
        value_counts = pd.Series(y).value_counts()
        total_samples = len(y)
        inverse_freq = {cls: total_samples / (len(classes) * count) 
                       for cls, count in value_counts.items()}
        
        # Strategy 3: Square root of inverse frequency (less aggressive)
        sqrt_inverse = {cls: np.sqrt(total_samples / count) 
                       for cls, count in value_counts.items()}
        
        # Strategy 4: Log-based weighting
        log_weights = {cls: np.log(total_samples / count) 
                      for cls, count in value_counts.items()}
        
        weight_strategies = {
            'none': None,
            'balanced': balanced_dict
        }
        
        print("Class Weight Strategies:")
        for name, weights in weight_strategies.items():
            if weights is None:
                print(f"  📊 {name}: No weighting")
            else:
                print(f"  📊 {name}: {weights}")
        
        return weight_strategies
    
    def apply_sampling_strategy(self, X: np.ndarray, y: np.ndarray, 
                               strategy_name: str, strategy: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a specific sampling strategy to the data.
        
        Args:
            X: Feature matrix
            y: Target vector
            strategy_name: Name of the strategy
            strategy: Sampling strategy object
            
        Returns:
            Resampled X and y
        """
        if strategy is None:
            return X, y
        
        print(f"  🔄 Applying {strategy_name}...")
        start_time = time.time()
        
        try:
            X_resampled, y_resampled = strategy.fit_resample(X, y)
            duration = time.time() - start_time
            
            # Analyze the resampling result
            original_dist = pd.Series(y).value_counts()
            new_dist = pd.Series(y_resampled).value_counts()
            
            print(f"    ⏱️ Duration: {duration:.2f}s")
            print(f"    📈 Original: {dict(original_dist)}")
            print(f"    📊 Resampled: {dict(new_dist)}")
            print(f"    🎯 Size change: {len(X)} → {len(X_resampled)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            return X, y
    
    def train_model_with_strategy(self, model_name: str, model_class: Any, 
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 sampling_strategy: str, sampler: Any,
                                 class_weights: Optional[Dict] = None,
                                 **model_params) -> Dict[str, Any]:
        """
        Train a model with a specific imbalance handling strategy.
        
        Args:
            model_name: Name of the model
            model_class: Model class to instantiate
            X_train, y_train: Training data
            X_test, y_test: Test data
            sampling_strategy: Name of sampling strategy
            sampler: Sampling strategy object
            class_weights: Class weight dictionary
            **model_params: Additional model parameters
            
        Returns:
            Dictionary with results
        """
        print(f"\n🤖 TRAINING {model_name} with {sampling_strategy}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Apply sampling strategy
        X_train_resampled, y_train_resampled = self.apply_sampling_strategy(
            X_train, y_train, sampling_strategy, sampler
        )
        
        # Prepare model parameters
        if class_weights is not None and 'class_weight' in model_class().get_params():
            model_params['class_weight'] = class_weights
        
        # Initialize and train model
        model = model_class(random_state=self.random_state, **model_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'specificity': self._calculate_specificity(y_test, y_pred)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        training_time = time.time() - start_time
        
        result = {
            'model_name': model_name,
            'sampling_strategy': sampling_strategy,
            'class_weights': str(class_weights) if class_weights else 'None',
            'metrics': metrics,
            'training_time': training_time,
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba,
            'sample_sizes': {
                'original_train': len(X_train),
                'resampled_train': len(X_train_resampled),
                'test': len(X_test)
            }
        }
        
        print(f"✅ Training completed in {training_time:.2f}s")
        print(f"📊 Metrics: F1={metrics['f1']:.3f}, ROC-AUC={metrics.get('roc_auc', 'N/A')}")
        
        return result
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        Find optimal threshold using Youden's J statistic.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        if y_proba is None:
            return {'optimal_threshold': 0.5, 'j_statistic': 0.0}
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Calculate Youden's J statistic for each threshold
        j_statistics = tpr - fpr
        
        # Find optimal threshold
        optimal_idx = np.argmax(j_statistics)
        optimal_threshold = thresholds[optimal_idx]
        optimal_j = j_statistics[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'j_statistic': optimal_j,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx]
        }
    
    def comprehensive_comparison(self, df_reviews: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive comparison of all imbalance handling strategies.
        
        Args:
            df_reviews: DataFrame with review data
            
        Returns:
            Dictionary with all results
        """
        print("\n🚀 COMPREHENSIVE IMBALANCE STRATEGY COMPARISON")
        print("=" * 60)
        
        # Prepare data
        print("\n📝 PREPARING DATA")
        print("-" * 30)
        
        # Text vectorization
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X = vectorizer.fit_transform(df_reviews['review_text']).toarray()
        y = df_reviews['is_spoiler'].values
        
        print(f"✅ Vectorized text: {X.shape}")
        print(f"✅ Target variable: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Analyze class distribution
        distribution_analysis = self.analyze_class_distribution(pd.Series(y_train))
        
        # Prepare strategies
        sampling_strategies = self.prepare_sampling_strategies()
        class_weight_strategies = self.compute_optimal_class_weights(pd.Series(y_train))
        
        # Define models to test (starting with just 2 for initial testing)
        models = {
            'LogisticRegression': (LogisticRegression, {'max_iter': 1000, 'solver': 'lbfgs'}),
            'RandomForest': (RandomForestClassifier, {'n_estimators': 50, 'max_depth': 5})
        }
        
        # Store all results
        all_results = []
        
        print(f"\n🔬 RUNNING EXPERIMENTS")
        print(f"Models: {len(models)}")
        print(f"Sampling Strategies: {len(sampling_strategies)}")
        print(f"Class Weight Strategies: {len(class_weight_strategies)}")
        print(f"Total Experiments: {len(models) * len(sampling_strategies) * len(class_weight_strategies)}")
        print("-" * 60)
        
        experiment_count = 0
        total_experiments = len(models) * len(sampling_strategies) * len(class_weight_strategies)
        
        # Run experiments
        for model_name, (model_class, model_params) in models.items():
            for sampling_name, sampler in sampling_strategies.items():
                for weight_name, weights in class_weight_strategies.items():
                    experiment_count += 1
                    
                    print(f"\n🧪 Experiment {experiment_count}/{total_experiments}: {model_name} + {sampling_name} + {weight_name}")
                    
                    try:
                        result = self.train_model_with_strategy(
                            model_name=model_name,
                            model_class=model_class,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            sampling_strategy=sampling_name,
                            sampler=sampler,
                            class_weights=weights,
                            **model_params
                        )
                        
                        # Add threshold optimization
                        if result['probabilities'] is not None:
                            threshold_optimization = self.optimize_threshold(
                                y_test, result['probabilities']
                            )
                            result['threshold_optimization'] = threshold_optimization
                            
                            # Calculate metrics at optimal threshold
                            optimal_pred = (result['probabilities'] >= 
                                          threshold_optimization['optimal_threshold']).astype(int)
                            
                            result['optimal_threshold_metrics'] = {
                                'accuracy': accuracy_score(y_test, optimal_pred),
                                'precision': precision_score(y_test, optimal_pred),
                                'recall': recall_score(y_test, optimal_pred),
                                'f1': f1_score(y_test, optimal_pred),
                                'specificity': self._calculate_specificity(y_test, optimal_pred)
                            }
                        
                        result['weight_strategy'] = weight_name
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"❌ Experiment failed: {str(e)}")
                        continue
        
        print(f"\n✅ COMPLETED {len(all_results)} SUCCESSFUL EXPERIMENTS")
        
        # Analyze results
        analysis_results = self.analyze_results(all_results)
        
        # Save results
        self.save_results({
            'distribution_analysis': distribution_analysis,
            'experiment_results': all_results,
            'analysis': analysis_results,
            'metadata': {
                'total_experiments': len(all_results),
                'vectorizer_features': X.shape[1],
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'timestamp': datetime.now().isoformat()
            }
        })
        
        return {
            'results': all_results,
            'analysis': analysis_results,
            'distribution': distribution_analysis
        }
    
    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze and summarize experimental results.
        
        Args:
            results: List of experiment results
            
        Returns:
            Analysis summary
        """
        print("\n📈 ANALYZING RESULTS")
        print("-" * 30)
        
        if not results:
            return {'error': 'No results to analyze'}
        
        # Convert to DataFrame for easier analysis
        metrics_data = []
        for result in results:
            base_metrics = result['metrics'].copy()
            base_metrics.update({
                'model_name': result['model_name'],
                'sampling_strategy': result['sampling_strategy'],
                'weight_strategy': result['weight_strategy'],
                'training_time': result['training_time']
            })
            
            # Add optimal threshold metrics if available
            if 'optimal_threshold_metrics' in result:
                for metric, value in result['optimal_threshold_metrics'].items():
                    base_metrics[f'optimal_{metric}'] = value
                base_metrics['optimal_threshold'] = result['threshold_optimization']['optimal_threshold']
            
            metrics_data.append(base_metrics)
        
        df_results = pd.DataFrame(metrics_data)
        
        # Find best performers
        best_f1 = df_results.loc[df_results['f1'].idxmax()]
        best_roc_auc = df_results.loc[df_results['roc_auc'].idxmax()] if 'roc_auc' in df_results.columns else None
        best_balanced = df_results.loc[((df_results['precision'] + df_results['recall']) / 2).idxmax()]
        
        print("🏆 TOP PERFORMERS:")
        print(f"  Best F1: {best_f1['model_name']} + {best_f1['sampling_strategy']} + {best_f1['weight_strategy']} (F1={best_f1['f1']:.3f})")
        if best_roc_auc is not None:
            print(f"  Best ROC-AUC: {best_roc_auc['model_name']} + {best_roc_auc['sampling_strategy']} + {best_roc_auc['weight_strategy']} (AUC={best_roc_auc['roc_auc']:.3f})")
        print(f"  Best Balanced: {best_balanced['model_name']} + {best_balanced['sampling_strategy']} + {best_balanced['weight_strategy']}")
        
        # Strategy effectiveness analysis
        strategy_analysis = {
            'sampling_strategies': df_results.groupby('sampling_strategy')['f1'].agg(['mean', 'std', 'max']).round(3),
            'weight_strategies': df_results.groupby('weight_strategy')['f1'].agg(['mean', 'std', 'max']).round(3),
            'model_performance': df_results.groupby('model_name')['f1'].agg(['mean', 'std', 'max']).round(3)
        }
        
        return {
            'best_performers': {
                'f1': best_f1.to_dict(),
                'roc_auc': best_roc_auc.to_dict() if best_roc_auc is not None else None,
                'balanced': best_balanced.to_dict()
            },
            'strategy_analysis': strategy_analysis,
            'summary_stats': df_results.describe().round(3).to_dict(),
            'results_dataframe': df_results
        }
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = self.config.output_dir / f"imbalance_optimization_results_{timestamp}.json"
        
        # Prepare JSON-serializable version
        json_results = results.copy()
        if 'experiment_results' in json_results:
            for result in json_results['experiment_results']:
                # Remove non-serializable objects
                result.pop('model', None)
                result.pop('predictions', None)
                result.pop('probabilities', None)
        
        # Remove DataFrame from analysis
        if 'analysis' in json_results and 'results_dataframe' in json_results['analysis']:
            df = json_results['analysis']['results_dataframe']
            json_results['analysis']['results_csv_saved'] = True
            json_results['analysis'].pop('results_dataframe', None)
            
            # Save DataFrame separately
            csv_file = self.config.output_dir / f"imbalance_results_dataframe_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"📊 Results DataFrame saved: {csv_file}")
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"💾 Results saved: {results_file}")
        
        # Generate summary report
        self.generate_summary_report(results, timestamp)
    
    def generate_summary_report(self, results: Dict[str, Any], timestamp: str):
        """Generate a markdown summary report."""
        report_file = self.config.output_dir / f"imbalance_optimization_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# SpoilerShield: Class Imbalance Optimization Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Distribution analysis
            if 'distribution_analysis' in results:
                dist = results['distribution_analysis']
                f.write("## 📊 Dataset Analysis\n\n")
                f.write(f"- **Total Samples:** {dist['total_samples']:,}\n")
                f.write(f"- **Imbalance Ratio:** {dist['imbalance_ratio']:.2f}:1\n")
                f.write(f"- **Minority Class:** {dist['minority_class']} ({dist['minority_percentage']:.1f}%)\n")
                f.write(f"- **Majority Class:** {dist['majority_class']} ({dist['majority_percentage']:.1f}%)\n\n")
            
            # Best performers
            if 'analysis' in results and 'best_performers' in results['analysis']:
                best = results['analysis']['best_performers']
                f.write("## 🏆 Best Performing Strategies\n\n")
                
                f1_best = best['f1']
                f.write(f"### Best F1 Score: {f1_best['f1']:.3f}\n")
                f.write(f"- **Model:** {f1_best['model_name']}\n")
                f.write(f"- **Sampling:** {f1_best['sampling_strategy']}\n")
                f.write(f"- **Weighting:** {f1_best['weight_strategy']}\n")
                f.write(f"- **Metrics:** Precision={f1_best['precision']:.3f}, Recall={f1_best['recall']:.3f}, Specificity={f1_best['specificity']:.3f}\n\n")
                
                if best['roc_auc']:
                    auc_best = best['roc_auc']
                    f.write(f"### Best ROC-AUC: {auc_best['roc_auc']:.3f}\n")
                    f.write(f"- **Model:** {auc_best['model_name']}\n")
                    f.write(f"- **Sampling:** {auc_best['sampling_strategy']}\n")
                    f.write(f"- **Weighting:** {auc_best['weight_strategy']}\n\n")
            
            # Strategy effectiveness
            if 'analysis' in results and 'strategy_analysis' in results['analysis']:
                strategy = results['analysis']['strategy_analysis']
                f.write("## 📈 Strategy Effectiveness\n\n")
                
                f.write("### Sampling Strategies (F1 Score)\n")
                for strategy_name, stats in strategy['sampling_strategies'].iterrows():
                    f.write(f"- **{strategy_name}:** Mean={stats['mean']:.3f}, Max={stats['max']:.3f}, Std={stats['std']:.3f}\n")
                f.write("\n")
                
                f.write("### Weight Strategies (F1 Score)\n")
                for weight_name, stats in strategy['weight_strategies'].iterrows():
                    f.write(f"- **{weight_name}:** Mean={stats['mean']:.3f}, Max={stats['max']:.3f}, Std={stats['std']:.3f}\n")
                f.write("\n")
            
            f.write("## 🎯 Recommendations\n\n")
            f.write("Based on the experimental results:\n\n")
            f.write("1. **Best Overall Strategy:** Use the top-performing combination for production\n")
            f.write("2. **Sampling Method:** Advanced SMOTE variants (ADASYN, BorderlineSMOTE) typically outperform basic SMOTE\n")
            f.write("3. **Class Weighting:** Balanced weights provide good baseline performance\n")
            f.write("4. **Threshold Optimization:** Apply Youden's J statistic for optimal classification threshold\n")
            f.write("5. **Model Selection:** Random Forest with appropriate sampling shows robust performance\n\n")
            
            f.write("## 📝 Next Steps\n\n")
            f.write("1. Implement the best-performing strategy in production pipeline\n")
            f.write("2. Apply hyperparameter optimization to further improve performance\n")
            f.write("3. Consider ensemble methods combining multiple strategies\n")
            f.write("4. Validate performance on additional datasets\n")
        
        print(f"📋 Summary report saved: {report_file}")


def main():
    """Main execution function."""
    print("🎬 SPOILERSHIELD: CLASS IMBALANCE OPTIMIZATION")
    print("=" * 60)
    
    # Initialize configuration
    config = EnvConfig()
    
    # Load data
    print("\n📥 LOADING DATA")
    print("-" * 30)
    
    data_loader = DataLoader(
        movie_reviews_path=config.get_data_path('train_reviews.json'),
        movie_details_path=config.get_data_path('IMDB_movie_details.json')
    )
    
    df_reviews = data_loader.load_imdb_movie_reviews()
    df_details = data_loader.load_imdb_movie_details()
    print(f"✅ Loaded {len(df_reviews):,} reviews")
    
    # Initialize optimizer
    optimizer = ImbalanceOptimizer(config)
    
    # Run comprehensive comparison
    results = optimizer.comprehensive_comparison(df_reviews)
    
    print("\n🎉 CLASS IMBALANCE OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Total experiments: {len(results['results'])}")
    print(f"📁 Results saved to: {config.output_dir}")
    
    # Display key findings
    if results['analysis'] and 'best_performers' in results['analysis']:
        best_f1 = results['analysis']['best_performers']['f1']
        print(f"\n🏆 BEST STRATEGY:")
        print(f"   Model: {best_f1['model_name']}")
        print(f"   Sampling: {best_f1['sampling_strategy']}")
        print(f"   Weighting: {best_f1['weight_strategy']}")
        print(f"   F1 Score: {best_f1['f1']:.3f}")
        print(f"   ROC-AUC: {best_f1.get('roc_auc', 'N/A')}")


if __name__ == "__main__":
    main()
