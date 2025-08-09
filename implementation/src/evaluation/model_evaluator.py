"""
Model evaluation utilities for comprehensive analysis and comparison.

This module provides tools for evaluating and comparing text classification models
using various metrics, visualizations, and statistical analyses.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison tool.
    
    This class provides methods for evaluating individual models and
    comparing multiple models using various metrics and visualizations.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation plots and results
        """
        self.output_dir = Path(output_dir) if output_dir else Path('outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store evaluation results
        self.model_results = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def find_threshold(self, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, int]:
        """
        Find the threshold that maximizes the Youden's J statistic.
        
        The Youden's J statistic is a measure of the overall performance of a
        classification model. It is defined as the sensitivity (tpr) minus the
        false positive rate (fpr).
        
        Args:
            fpr: A numpy array of false positive rates.
            tpr: A numpy array of true positive rates.
            thresholds: A numpy array of thresholds.
            
        Returns:
            th_opt: A float representing the threshold that maximizes the Youden's J statistic.
            indx_opt: An integer representing the index of the threshold that maximizes the Youden's J statistic.
        """
        j_scores = tpr - fpr
        indx_opt = np.argmax(j_scores)
        th_opt = thresholds[indx_opt]
        return th_opt, indx_opt
    
    def analyze_optimal_threshold(self, model_name: str, 
                                y_true: np.ndarray, 
                                y_proba: np.ndarray,
                                save_plot: bool = True) -> Dict[str, Any]:
        """
        Analyze and visualize optimal threshold using Youden's J statistic.
        
        Args:
            model_name: Name of the model for identification
            y_true: True labels
            y_proba: Prediction probabilities
            save_plot: Whether to save the plot
            
        Returns:
            Dict containing optimal threshold analysis results
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Find optimal threshold
        th_opt, opt_indx = self.find_threshold(fpr, tpr, thresholds)
        
        # Calculate metrics at optimal threshold
        y_pred_opt = (y_proba >= th_opt).astype(int)
        metrics_opt = {
            'accuracy': accuracy_score(y_true, y_pred_opt),
            'precision': precision_score(y_true, y_pred_opt),
            'recall': recall_score(y_true, y_pred_opt),
            'f1': f1_score(y_true, y_pred_opt),
            'threshold': th_opt,
            'youden_j': tpr[opt_indx] - fpr[opt_indx]
        }
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_proba):.3f})')
        plt.plot(fpr[opt_indx], tpr[opt_indx], 'ro', markersize=10, 
                label=f'Optimal threshold = {th_opt:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve with Optimal Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with optimal threshold metrics
        textstr = f'Optimal Threshold: {th_opt:.3f}\n'
        textstr += f'Youden\'s J: {metrics_opt["youden_j"]:.3f}\n'
        textstr += f'Accuracy: {metrics_opt["accuracy"]:.3f}\n'
        textstr += f'Precision: {metrics_opt["precision"]:.3f}\n'
        textstr += f'Recall: {metrics_opt["recall"]:.3f}\n'
        textstr += f'F1: {metrics_opt["f1"]:.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        if save_plot:
            plot_path = self.output_dir / f'{model_name}_optimal_threshold.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Optimal threshold analysis saved to: {plot_path}")
        
        plt.show()
        
        # Print results
        print(f"\nOptimal Threshold Analysis for {model_name}:")
        print(f"Optimal threshold: {th_opt:.3f}")
        print(f"Youden's J statistic: {metrics_opt['youden_j']:.3f}")
        print(f"Metrics at optimal threshold:")
        for metric, value in metrics_opt.items():
            if metric not in ['threshold', 'youden_j']:
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        return metrics_opt
    
    def evaluate_model(self, 
                      model_name: str,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_proba: Optional[np.ndarray] = None,
                      save_plots: bool = True) -> Dict[str, Any]:
        """
        Evaluate a single model and store results.
        
        Args:
            model_name: Name of the model for identification
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            save_plots: Whether to save evaluation plots
            
        Returns:
            Dict containing evaluation results
        """
        print(f"Evaluating {model_name}...")
        
        # Confusion matrix (calculate first for specificity)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
        
        # Calculate specificity (true negative rate) from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Brief debug info for specificity issues
        if metrics['specificity'] == 0.0:
            print(f"⚠️  {model_name}: Specificity = 0.0 (TN={tn}, FP={fp}, all negatives predicted as positive)")
            # Show class distribution for debugging
            unique, counts = np.unique(y_pred, return_counts=True)
            pred_dist = dict(zip(unique, counts))
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            true_dist = dict(zip(unique_true, counts_true))
            print(f"    True labels distribution: {true_dist}")
            print(f"    Predicted labels distribution: {pred_dist}")
            
            # Show probability distribution if available
            if y_proba is not None:
                print(f"    Probability stats: min={y_proba.min():.3f}, max={y_proba.max():.3f}, mean={y_proba.mean():.3f}")
                print(f"    Probabilities > 0.5: {(y_proba > 0.5).sum()}/{len(y_proba)} ({(y_proba > 0.5).mean()*100:.1f}%)")
        elif metrics['specificity'] < 0.1:
            print(f"⚠️  {model_name}: Low specificity = {metrics['specificity']:.4f} (TN={tn}, FP={fp})")
        
        # Add probability-based metrics if available
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_proba)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Store results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        # Generate curves if probabilities are available
        if y_proba is not None:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            
            results['roc_curve'] = {
                'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds
            }
            results['pr_curve'] = {
                'precision': precision, 'recall': recall, 'thresholds': pr_thresholds
            }
        
        # Store in evaluator
        self.model_results[model_name] = results
        
        # Print results with proper display names
        print(f"Results for {model_name}:")
        
        # Use same metric display names as plots and reports
        metric_names = {
            'accuracy': 'Accuracy',
            'precision': 'Precision', 
            'recall': 'Recall (Sensitivity)',
            'specificity': 'Specificity',
            'f1': 'F1 Score',
            'roc_auc': 'ROC-AUC',
            'average_precision': 'Average Precision'
        }
        
        for metric, value in metrics.items():
            display_name = metric_names.get(metric, metric.capitalize())
            print(f"  {display_name}: {value:.4f}")
        
        # Save plots if requested
        if save_plots:
            self._save_model_plots(model_name, results)
        
        # For imbalanced data, also calculate metrics at optimal threshold
        if y_proba is not None and metrics['specificity'] < 0.5:
            print(f"🔧 Computing optimal threshold for {model_name} (current specificity = {metrics['specificity']:.3f})...")
            
            # Find threshold that balances precision and recall  
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = pr_thresholds[optimal_idx]
            
            # Calculate metrics at optimal threshold
            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
            cm_optimal = confusion_matrix(y_true, y_pred_optimal)
            tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()
            
            optimal_metrics = {
                'accuracy': accuracy_score(y_true, y_pred_optimal),
                'precision': precision_score(y_true, y_pred_optimal),
                'recall': recall_score(y_true, y_pred_optimal),
                'f1': f1_score(y_true, y_pred_optimal),
                'specificity': tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0,
                'threshold': optimal_threshold
            }
            
            print(f"    📊 Optimal threshold: {optimal_threshold:.3f}")
            print(f"    📊 Improved specificity: {optimal_metrics['specificity']:.3f} (vs {metrics['specificity']:.3f})")
            print(f"    📊 F1 at optimal threshold: {optimal_metrics['f1']:.3f} (vs {metrics['f1']:.3f})")
            
            # Store optimal metrics in results
            results['optimal_threshold_metrics'] = optimal_metrics

        return results
    
    def evaluate_with_custom_threshold(self, 
                                     model_name: str,
                                     y_true: np.ndarray,
                                     y_proba: np.ndarray,
                                     threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate model performance with a custom probability threshold.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_proba: Prediction probabilities
            threshold: Custom threshold for converting probabilities to predictions
            
        Returns:
            Dict with metrics at the custom threshold
        """
        # Convert probabilities to predictions using custom threshold
        y_pred_custom = (y_proba >= threshold).astype(int)
        
        # Calculate all metrics
        cm = confusion_matrix(y_true, y_pred_custom)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred_custom),
            'precision': precision_score(y_true, y_pred_custom),
            'recall': recall_score(y_true, y_pred_custom),
            'f1': f1_score(y_true, y_pred_custom),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        print(f"📊 {model_name} at threshold {threshold:.3f}:")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall (Sensitivity): {metrics['recall']:.3f}")
        print(f"   Specificity: {metrics['specificity']:.3f}")
        print(f"   F1 Score: {metrics['f1']:.3f}")
        print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        return metrics
    
    def compare_models(self, 
                      metrics_to_compare: Optional[List[str]] = None,
                      save_plots: bool = True) -> pd.DataFrame:
        """
        Compare multiple models using specified metrics.
        
        Args:
            metrics_to_compare: List of metrics to compare (default: all available)
            save_plots: Whether to save comparison plots
            
        Returns:
            DataFrame with comparison results
        """
        if not self.model_results:
            raise ValueError("No models have been evaluated yet.")
        
        # Default metrics to compare
        if metrics_to_compare is None:
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Extract metrics for each model
        comparison_data = []
        for model_name, results in self.model_results.items():
            model_metrics = results['metrics']
            row = {'model': model_name}
            for metric in metrics_to_compare:
                row[metric] = model_metrics.get(metric, np.nan)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison
        print("\nModel Comparison:")
        print("=" * 50)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison plots if requested
        if save_plots:
            self._save_comparison_plots(comparison_df, metrics_to_compare)
        
        return comparison_df
    
    def plot_confusion_matrices(self, 
                               figsize: Tuple[int, int] = (15, 5),
                               save_plot: bool = True) -> None:
        """
        Plot confusion matrices for all evaluated models.
        
        Args:
            figsize: Figure size for the plot
            save_plot: Whether to save the plot
        """
        n_models = len(self.model_results)
        if n_models == 0:
            print("No models to plot.")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.model_results.items()):
            cm = results['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Spoiler', 'Spoiler'],
                       yticklabels=['No Spoiler', 'Spoiler'],
                       ax=axes[i])
            
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'confusion_matrices.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to: {plot_path}")
        
        plt.show()
    
    def plot_roc_curves(self, 
                       figsize: Tuple[int, int] = (10, 8),
                       save_plot: bool = True,
                       show_optimal_threshold: bool = True) -> None:
        """
        Plot ROC curves for all models with probabilities.
        
        Args:
            figsize: Figure size for the plot
            save_plot: Whether to save the plot
            show_optimal_threshold: Whether to show optimal threshold points
        """
        models_with_proba = {
            name: results for name, results in self.model_results.items()
            if 'roc_curve' in results
        }
        
        if not models_with_proba:
            print("No models with probability predictions to plot ROC curves.")
            return
        
        plt.figure(figsize=figsize)
        
        # Use distinct, easily distinguishable colors
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Plot ROC curve for each model
        for i, (model_name, results) in enumerate(models_with_proba.items()):
            roc_data = results['roc_curve']
            auc = results['metrics']['roc_auc']
            color = distinct_colors[i % len(distinct_colors)]
            
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    color=color, linewidth=2,
                    label=f'{model_name} (AUC = {auc:.3f})')
            
            # Add optimal threshold point if requested - SAME COLOR as curve
            if show_optimal_threshold:
                th_opt, opt_indx = self.find_threshold(
                    roc_data['fpr'], roc_data['tpr'], roc_data['thresholds']
                )
                plt.plot(roc_data['fpr'][opt_indx], roc_data['tpr'][opt_indx], 
                        'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1,
                        label=f'{model_name} Optimal (t={th_opt:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'roc_curves.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {plot_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, 
                                   figsize: Tuple[int, int] = (15, 10),
                                   save_plot: bool = True) -> None:
        """
        Plot Precision and Recall vs Threshold curves in separate subplots for each model.
        
        Shows precision and recall as functions of threshold in individual subplots,
        with crossover points where precision ≈ recall marked for each model.
        
        Args:
            figsize: Figure size for the plot
            save_plot: Whether to save the plot
        """
        models_with_proba = {
            name: results for name, results in self.model_results.items()
            if 'pr_curve' in results
        }
        
        if not models_with_proba:
            print("No models with probability predictions to plot PR curves.")
            return
        
        n_models = len(models_with_proba)
        # Calculate subplot layout (prefer wider layout)
        cols = min(3, n_models)  # Max 3 columns
        rows = (n_models + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Use distinct, easily distinguishable colors (same as ROC curves)
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Plot precision and recall vs threshold for each model in separate subplots
        for i, (model_name, results) in enumerate(models_with_proba.items()):
            ax = axes[i]
            pr_data = results['pr_curve']
            precision = pr_data['precision']
            recall = pr_data['recall']
            thresholds = pr_data['thresholds']
            
            # Handle threshold array length
            precision_plot = precision[:-1] if len(precision) > len(thresholds) else precision
            recall_plot = recall[:-1] if len(recall) > len(thresholds) else recall
            
            color = distinct_colors[i % len(distinct_colors)]
            
            # Plot precision vs threshold
            ax.plot(thresholds, precision_plot, 
                   color=color, linestyle='-', linewidth=2,
                   label='Precision')
            
            # Plot recall vs threshold  
            ax.plot(thresholds, recall_plot,
                   color=color, linestyle='--', linewidth=2,
                   label='Recall (Sensitivity)')
            
            # Find crossover point where |precision - recall| is minimal
            diff = np.abs(precision_plot - recall_plot)
            crossover_idx = np.argmin(diff)
            crossover_threshold = thresholds[crossover_idx]
            crossover_precision = precision_plot[crossover_idx]
            crossover_recall = recall_plot[crossover_idx]
            
            # Plot crossover point
            ax.plot(crossover_threshold, crossover_precision, 
                   'o', color=color, markersize=8, markeredgecolor='black', 
                   markeredgewidth=1,
                   label=f'Crossover (t={crossover_threshold:.3f})')
            
            # Format subplot
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title(f'{model_name}')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)
        
        # Overall title
        fig.suptitle('Precision and Recall vs Threshold with Crossover Points', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'precision_recall_vs_threshold.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall vs Threshold curves saved to: {plot_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, 
                              metrics: List[str] = None,
                              figsize: Tuple[int, int] = (12, 8),
                              save_plot: bool = True) -> None:
        """
        Create a bar plot comparing metrics across models.
        
        Args:
            metrics: List of metrics to compare
            figsize: Figure size for the plot
            save_plot: Whether to save the plot
        """
        if not self.model_results:
            print("No models to compare.")
            return
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']
        
        # Create metric display name mapping
        metric_names = {
            'accuracy': 'Accuracy',
            'precision': 'Precision', 
            'recall': 'Recall (Sensitivity)',
            'specificity': 'Specificity',
            'f1': 'F1 Score',
            'roc_auc': 'ROC-AUC',
            'average_precision': 'Average Precision'
        }
        
        # Prepare data for plotting
        plot_data = []
        for model_name, results in self.model_results.items():
            for metric in metrics:
                value = results['metrics'].get(metric, np.nan)
                if not np.isnan(value):
                    display_name = metric_names.get(metric, metric.capitalize())
                    plot_data.append({
                        'Model': model_name,
                        'Metric': display_name,
                        'Value': value
                    })
        
        if not plot_data:
            print("No valid metrics to plot.")
            return
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Create grouped bar plot
        pivot_df = df_plot.pivot(index='Model', columns='Metric', values='Value')
        pivot_df.plot(kind='bar', ax=plt.gca())
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'metrics_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to: {plot_path}")
        
        plt.show()
    
    def save_evaluation_report(self, filename: str = 'evaluation_report.txt') -> None:
        """
        Save a comprehensive evaluation report to a text file.
        
        Args:
            filename: Name of the report file
        """
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, results in self.model_results.items():
                f.write(f"MODEL: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                # Write metrics with proper display names
                f.write("Metrics:\n")
                
                # Use same metric display names as plots
                metric_names = {
                    'accuracy': 'Accuracy',
                    'precision': 'Precision', 
                    'recall': 'Recall (Sensitivity)',
                    'specificity': 'Specificity',
                    'f1': 'F1 Score',
                    'roc_auc': 'ROC-AUC',
                    'average_precision': 'Average Precision'
                }
                
                # Display in preferred order
                metric_order = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc', 'average_precision']
                
                for metric in metric_order:
                    if metric in results['metrics']:
                        display_name = metric_names.get(metric, metric.capitalize())
                        value = results['metrics'][metric]
                        f.write(f"  {display_name}: {value:.4f}\n")
                
                # Add any other metrics not in the standard order
                for metric, value in results['metrics'].items():
                    if metric not in metric_order:
                        display_name = metric_names.get(metric, metric.capitalize())
                        f.write(f"  {display_name}: {value:.4f}\n")
                
                f.write("\nClassification Report:\n")
                report = results['classification_report']
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict):
                        f.write(f"  {class_name}:\n")
                        for metric, value in metrics.items():
                            if isinstance(value, float):
                                f.write(f"    {metric}: {value:.4f}\n")
                            else:
                                f.write(f"    {metric}: {value}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"Evaluation report saved to: {report_path}")
    
    def _save_model_plots(self, model_name: str, results: Dict[str, Any]) -> None:
        """Save individual model plots."""
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Spoiler', 'Spoiler'],
                   yticklabels=['No Spoiler', 'Spoiler'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plot_path = self.output_dir / f'{model_name}_confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curve if available
        if 'roc_curve' in results:
            plt.figure(figsize=(8, 6))
            roc_data = results['roc_curve']
            auc = results['metrics']['roc_auc']
            
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'AUC = {auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            
            # Add optimal threshold point
            th_opt, opt_indx = self.find_threshold(
                roc_data['fpr'], roc_data['tpr'], roc_data['thresholds']
            )
            plt.plot(roc_data['fpr'][opt_indx], roc_data['tpr'][opt_indx], 
                    'ro', markersize=8, label=f'Optimal (t={th_opt:.3f})')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = self.output_dir / f'{model_name}_roc_curve.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_comparison_plots(self, comparison_df: pd.DataFrame, 
                             metrics: List[str]) -> None:
        """Save comparison plots."""
        # Metrics comparison bar plot
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        plot_data = []
        for _, row in comparison_df.iterrows():
            for metric in metrics:
                if not pd.isna(row[metric]):
                    plot_data.append({
                        'Model': row['model'],
                        'Metric': metric.capitalize(),
                        'Value': row[metric]
                    })
        
        df_plot = pd.DataFrame(plot_data)
        pivot_df = df_plot.pivot(index='Model', columns='Metric', values='Value')
        pivot_df.plot(kind='bar', ax=plt.gca())
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.output_dir / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close() 