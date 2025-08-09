"""
02_model_training.py - Model Training and Hyperparameter Search

This script implements milestones 3 and 4 of the SpoilerShield project:
3. Model Training and Hyperparameter Search
4. Model Evaluation

It loads the processed data from 01_data_eda.py and trains multiple models
with hyperparameter optimization, starting with the baseline TF-IDF + Logistic Regression.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Transformer imports (will be available when needed)
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers datasets")

# Add the src directory to the path for imports
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from models.baseline_model import BaselineModel
from evaluation.model_evaluator import ModelEvaluator
from eda.data_loader import DataLoader
from utils.env_config import EnvConfig

# Scikit-learn imports for hyperparameter search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)


def load_processed_data(config: EnvConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load processed data from 01_data_eda.py output.
    
    Args:
        config: Environment configuration
        
    Returns:
        Tuple of (processed_data, metadata)
    """
    print("=" * 60)
    print("1. LOADING PROCESSED DATA")
    print("=" * 60)
    
    # Load processed data
    data_path = config.output_dir / 'processed_data.pt'
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}. Please run 01_data_eda.py first.")
    
    processed_data = torch.load(data_path)
    print(f"✅ Loaded processed data from: {data_path}")
    
    # Load metadata
    metadata_path = config.output_dir / 'data_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata from: {metadata_path}")
    else:
        metadata = {}
        print("⚠️ No metadata file found")
    
    # Display data info
    print(f"\n📊 Data Information:")
    print(f"  Model: {processed_data['model_name']}")
    print(f"  Max Length: {processed_data['max_length']}")
    print(f"  Number of Samples: {processed_data['num_samples']}")
    print(f"  Input IDs Shape: {processed_data['input_ids'].shape}")
    print(f"  Attention Mask Shape: {processed_data['attention_mask'].shape}")
    print(f"  Labels Shape: {processed_data['labels'].shape}")
    
    return processed_data, metadata


def load_raw_data_for_baseline(config: EnvConfig) -> pd.DataFrame:
    """
    Load raw data for baseline model training (TF-IDF needs raw text).
    
    Args:
        config: Environment configuration
        
    Returns:
        DataFrame with raw text data
    """
    print("\n📄 Loading raw data for baseline model...")
    
    # Load raw data
    data_loader = DataLoader(
        config.get_data_path('train_reviews.json'),
        config.get_data_path('IMDB_movie_details.json')
    )
    
    df_reviews = data_loader.load_imdb_movie_reviews()
    df_reviews = df_reviews.dropna(subset=['review_text', 'is_spoiler'])
    
    print(f"✅ Loaded {len(df_reviews)} reviews for baseline training")
    
    return df_reviews


def train_baseline_model(df_reviews: pd.DataFrame, config: EnvConfig) -> Dict[str, Any]:
    """
    Train the baseline TF-IDF + Logistic Regression model.
    
    Args:
        df_reviews: DataFrame with review text and labels
        config: Environment configuration
        
    Returns:
        Dictionary with baseline model results
    """
    print("\n" + "=" * 60)
    print("2. BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Initialize baseline model
    baseline_model = BaselineModel(
        max_features=10000,
        ngram_range=(1, 2),
        random_state=42,
        test_size=0.2
    )
    
    # Prepare data
    baseline_model.prepare_data(
        texts=df_reviews['review_text'],
        labels=df_reviews['is_spoiler']
    )
    
    # Train the model
    train_metrics = baseline_model.train()
    
    # Evaluate the model
    eval_results = baseline_model.evaluate()
    
    # Cross-validation
    cv_results = baseline_model.cross_validate(cv_folds=5)
    
    # Get feature importance
    feature_importance = baseline_model.get_feature_importance(top_n=20)
    
    # Save the model
    model_path = config.output_dir / 'baseline_model.pkl'
    baseline_model.save_model(str(model_path))
    
    print(f"✅ Baseline model saved to: {model_path}")
    
    return {
        'model': baseline_model,
        'train_metrics': train_metrics,
        'eval_results': eval_results,
        'cv_results': cv_results,
        'feature_importance': feature_importance,
        'model_path': str(model_path)
    }


def hyperparameter_search_baseline(df_reviews: pd.DataFrame, config: EnvConfig) -> Dict[str, Any]:
    """
    Perform hyperparameter search for the baseline model.
    
    Args:
        df_reviews: DataFrame with review text and labels
        config: Environment configuration
        
    Returns:
        Dictionary with hyperparameter search results
    """
    print("\n" + "=" * 60)
    print("3. HYPERPARAMETER SEARCH - BASELINE")
    print("=" * 60)
    
    # Prepare data
    X = df_reviews['review_text']
    y = df_reviews['is_spoiler']
    
    # Define more conservative parameter grid to avoid numerical issues
    param_grid = {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [2, 5],
        'tfidf__max_df': [0.9, 0.95],
        'classifier__C': [1.0, 10.0, 100.0],  # Avoid very small C values
        'classifier__solver': ['lbfgs']  # Use more stable solver
    }
    
    print(f"🔍 Parameter grid size: {len(param_grid['tfidf__max_features']) * len(param_grid['tfidf__ngram_range']) * len(param_grid['tfidf__min_df']) * len(param_grid['tfidf__max_df']) * len(param_grid['classifier__C']) * len(param_grid['classifier__solver'])} combinations")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=2000,  # Increase iterations for better convergence
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduce CV folds for speed
    
    # Suppress all warnings during grid search to clean up output
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    
    # Comprehensive warning suppression
    warnings.filterwarnings('ignore')  # Suppress all warnings
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Also suppress numpy warnings
    import numpy as np
    np.seterr(all='ignore')
    
    # Perform grid search
    print("🔍 Performing grid search...")
    print("⏱️  This may take several minutes...")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        error_score='raise'  # Raise errors instead of silently failing
    )
    
    try:
        grid_search.fit(X, y)
        print("✅ Grid search completed successfully!")
    except Exception as e:
        print(f"❌ Grid search failed with error: {e}")
        # Fallback to simpler parameter grid
        print("🔄 Trying with simpler parameter grid...")
        simple_param_grid = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [1.0, 10.0]
        }
        grid_search = GridSearchCV(
            pipeline,
            simple_param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=1,  # Single job to avoid parallelization issues
            verbose=1
        )
        grid_search.fit(X, y)
        print("✅ Fallback grid search completed!")
    
    # Reset warnings and numpy error handling
    warnings.resetwarnings()
    np.seterr(all='warn')  # Restore numpy warnings
    
    print(f"✅ Grid search completed!")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Save results
    results_path = config.output_dir / 'baseline_hyperparameter_search.json'
    results = {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
            'params': grid_search.cv_results_['params']
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Hyperparameter search results saved to: {results_path}")
    
    return {
        'grid_search': grid_search,
        'best_model': grid_search.best_estimator_,
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'results_path': str(results_path)
    }


def train_additional_models(df_reviews: pd.DataFrame, config: EnvConfig) -> Dict[str, Any]:
    """
    Train additional models for comparison.
    
    Args:
        df_reviews: DataFrame with review text and labels
        config: Environment configuration
        
    Returns:
        Dictionary with additional model results
    """
    print("\n" + "=" * 60)
    print("4. ADDITIONAL MODELS TRAINING")
    print("=" * 60)
    
    # Prepare data
    X = df_reviews['review_text']
    y = df_reviews['is_spoiler']
    
    # Create TF-IDF vectorizer with conservative settings
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        lowercase=True,
        strip_accents='unicode'
    )
    
    # Transform data
    print("🔄 Transforming text data with TF-IDF...")
    X_tfidf = tfidf.fit_transform(X)
    print(f"✅ TF-IDF transformation completed - Shape: {X_tfidf.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models to train with conservative settings and class imbalance handling
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50,  # Reduced for faster training
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'  # Better for RF with imbalanced data
        ),
        'SVM': SVC(
            kernel='linear',  # Linear kernel is much faster than RBF
            C=1.0,  # Conservative C value
            probability=True,
            random_state=42,
            class_weight='balanced',
            max_iter=1000  # Limit iterations
        ),
        'Naive Bayes': MultinomialNB(
            alpha=1.0  # Laplace smoothing
        )
    }
    
    results = {}
    
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    for model_name, model in models.items():
        print(f"\n🤖 Training {model_name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None
            }
            
            # Save model
            model_path = config.output_dir / f'{model_name.lower().replace(" ", "_")}_model.pkl'
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'tfidf': tfidf,
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'y_test': y_test
                }, f)
            
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'model_path': str(model_path),
                'y_pred': y_pred,
                'y_prob': y_prob, 
                'y_test': y_test
            }
            
            print(f"✅ {model_name} trained successfully!")
            roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
            print(f"   📊 F1: {metrics['f1']:.4f} | Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {roc_auc_str}")
            
        except Exception as e:
            print(f"❌ {model_name} training failed: {e}")
            print(f"   ⚠️  Skipping {model_name} and continuing...")
            continue
    
    # Reset warnings (no numpy reset needed here since we didn't set numpy errors)
    warnings.resetwarnings()
    
    print(f"\n✅ Additional models training completed! {len(results)} models trained successfully.")
    return results


def comprehensive_model_evaluation(baseline_results: Dict[str, Any], 
                                  additional_results: Dict[str, Any],
                                  config: EnvConfig) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of all models.
    
    Args:
        baseline_results: Results from baseline model
        additional_results: Results from additional models
        config: Environment configuration
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    print("\n" + "=" * 60)
    print("5. COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=str(config.output_dir))
    
    # Collect all results
    all_results = {}
    
    # Get baseline model data
    baseline_model = baseline_results['model']
    
    # Evaluate baseline model with ModelEvaluator
    evaluator.evaluate_model(
        model_name='Baseline_TF-IDF_LogReg',
        y_true=baseline_model.y_test,
        y_pred=baseline_results['eval_results']['predictions'],
        y_proba=baseline_results['eval_results']['probabilities'],
        save_plots=True
    )
    
    # Add baseline to all_results
    all_results['Baseline_TF-IDF_LogReg'] = {
        'y_true': baseline_model.y_test,
        'y_pred': baseline_results['eval_results']['predictions'],
        'y_prob': baseline_results['eval_results']['probabilities'],
        'metrics': baseline_results['eval_results']['metrics']
    }
    
    # Now evaluate ALL additional models with ModelEvaluator (this will calculate specificity!)
    for model_name, model_data in additional_results.items():
        if 'y_pred' in model_data and 'y_test' in model_data:
            print(f"\n🔍 Evaluating {model_name} with full metrics...")
            
            # Use ModelEvaluator to get complete metrics including specificity
            evaluator.evaluate_model(
                model_name=model_name,
                y_true=model_data['y_test'],
                y_pred=model_data['y_pred'],
                y_proba=model_data['y_prob'],
                save_plots=True
            )
            
            # Add to all_results for comparison
            all_results[model_name] = {
                'y_true': model_data['y_test'],
                'y_pred': model_data['y_pred'],
                'y_prob': model_data['y_prob'],
                'metrics': model_data['metrics']
            }
        else:
            print(f"⚠️ {model_name}: Missing prediction data, using stored metrics only")
            all_results[model_name] = {
                'metrics': model_data['metrics']
            }
    
    # Generate comparison plots
    evaluator.plot_confusion_matrices(save_plot=True)
    evaluator.plot_roc_curves(save_plot=True)
    evaluator.plot_precision_recall_curves(save_plot=True)
    evaluator.plot_metrics_comparison(save_plot=True)
    
    # Save comprehensive evaluation report
    report_path = config.output_dir / 'comprehensive_evaluation_report.txt'
    evaluator.save_evaluation_report(str(report_path))
    
    # Create model comparison summary using enhanced metrics from ModelEvaluator
    comparison_summary = {
        'models_evaluated': list(evaluator.model_results.keys()),
        'best_model': None,
        'best_f1_score': 0,
        'model_rankings': [],
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Find best model using the enhanced metrics from ModelEvaluator (includes specificity!)
    for model_name, results in evaluator.model_results.items():
        if 'metrics' in results:
            f1_score = results['metrics']['f1']
            if f1_score > comparison_summary['best_f1_score']:
                comparison_summary['best_f1_score'] = f1_score
                comparison_summary['best_model'] = model_name
            
            comparison_summary['model_rankings'].append({
                'model': model_name,
                'f1_score': f1_score,
                'accuracy': results['metrics'].get('accuracy', 0),
                'precision': results['metrics'].get('precision', 0),
                'recall': results['metrics'].get('recall', 0),
                'specificity': results['metrics'].get('specificity', 0),  # Now includes real specificity!
                'roc_auc': results['metrics'].get('roc_auc', None)
            })
    
    # Sort by F1 score
    comparison_summary['model_rankings'].sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Save comparison summary
    summary_path = config.output_dir / 'model_comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    print(f"✅ Comprehensive evaluation completed!")
    print(f"Best model: {comparison_summary['best_model']} (F1: {comparison_summary['best_f1_score']:.4f})")
    print(f"Evaluation report saved to: {report_path}")
    print(f"Comparison summary saved to: {summary_path}")
    
    return comparison_summary


def save_training_results(baseline_results: Dict[str, Any],
                         hyperparameter_results: Dict[str, Any],
                         additional_results: Dict[str, Any],
                         evaluation_summary: Dict[str, Any],
                         config: EnvConfig) -> None:
    """
    Save all training results to a comprehensive summary.
    
    Args:
        baseline_results: Baseline model results
        hyperparameter_results: Hyperparameter search results
        additional_results: Additional model results
        evaluation_summary: Comprehensive evaluation summary
        config: Environment configuration
    """
    print("\n" + "=" * 60)
    print("6. SAVING TRAINING RESULTS")
    print("=" * 60)
    
    # Create comprehensive summary
    training_summary = {
        'training_timestamp': datetime.now().isoformat(),
        'baseline_model': {
            'train_metrics': baseline_results['train_metrics'],
            'eval_metrics': baseline_results['eval_results']['metrics'],
            'cv_results': baseline_results['cv_results'],
            'model_path': baseline_results['model_path']
        },
        'hyperparameter_search': {
            'best_score': hyperparameter_results['best_score'],
            'best_params': hyperparameter_results['best_params'],
            'results_path': hyperparameter_results['results_path']
        },
        'additional_models': {
            model_name: {
                'metrics': model_data['metrics'],
                'model_path': model_data['model_path']
            }
            for model_name, model_data in additional_results.items()
        },
        'evaluation_summary': evaluation_summary,
        'files_created': []
    }
    
    # Add file paths
    training_summary['files_created'].extend([
        baseline_results['model_path'],
        hyperparameter_results['results_path']
    ])
    
    for model_data in additional_results.values():
        training_summary['files_created'].append(model_data['model_path'])
    
    # Save training summary
    summary_path = config.output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    
    print(f"✅ Training summary saved to: {summary_path}")
    
    # Create training report
    report_path = config.output_dir / 'training_report.md'
    with open(report_path, 'w') as f:
        f.write("# SpoilerShield Model Training Report\n\n")
        f.write(f"**Training Date:** {training_summary['training_timestamp']}\n\n")
        
        f.write("## Model Performance Summary\n\n")
        f.write("| Model | F1 Score | Accuracy | Precision | Recall (Sensitivity) | Specificity | ROC-AUC |\n")
        f.write("|-------|----------|----------|-----------|---------------------|-------------|--------|\n")
        
        for model_info in evaluation_summary['model_rankings']:
            specificity = model_info.get('specificity', 'N/A')
            roc_auc = model_info.get('roc_auc', 'N/A')
            
            # Format specificity and roc_auc values
            specificity_str = f"{specificity:.4f}" if isinstance(specificity, (int, float)) else specificity
            roc_auc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else roc_auc
            
            f.write(f"| {model_info['model']} | {model_info['f1_score']:.4f} | "
                   f"{model_info['accuracy']:.4f} | {model_info['precision']:.4f} | "
                   f"{model_info['recall']:.4f} | "
                   f"{specificity_str} | "
                   f"{roc_auc_str} |\n")
        
        f.write(f"\n**Best Model:** {evaluation_summary['best_model']}\n")
        f.write(f"**Best F1 Score:** {evaluation_summary['best_f1_score']:.4f}\n\n")
        
        f.write("## Hyperparameter Search Results\n\n")
        f.write(f"**Best Parameters:** {hyperparameter_results['best_params']}\n")
        f.write(f"**Best Cross-Validation Score:** {hyperparameter_results['best_score']:.4f}\n\n")
        
        f.write("## Files Created\n\n")
        for file_path in training_summary['files_created']:
            f.write(f"- {file_path}\n")
    
    print(f"✅ Training report saved to: {report_path}")


# ============================================================
# TRANSFORMER MODEL TRAINING FUNCTIONS (NEW)
# ============================================================

def setup_transformer_environment() -> bool:
    """
    Check and setup transformer training environment.
    
    Returns:
        bool: True if transformers are available and GPU is detected
    """
    print("\n" + "=" * 60)
    print("🤖 TRANSFORMER ENVIRONMENT SETUP")
    print("=" * 60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available")
        print("   Install with: pip install transformers datasets accelerate")
        return False
    
    print("✅ Transformers library available")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print("🚀 MPS (Apple Silicon) available")
    else:
        print("⚠️ No GPU detected - will use CPU (much slower)")
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"📱 Using device: {device}")
    
    return True


def prepare_transformer_data(df_reviews: pd.DataFrame, 
                           model_name: str = "microsoft/deberta-v3-base",
                           max_length: int = 512,
                           test_size: float = 0.2) -> Dict[str, Any]:
    """
    Prepare data for transformer training.
    
    Args:
        df_reviews: DataFrame with review text and labels
        model_name: Hugging Face model name
        max_length: Maximum sequence length
        test_size: Test split ratio
        
    Returns:
        Dict with tokenized datasets and tokenizer
    """
    print(f"\n🔤 PREPARING DATA FOR {model_name}")
    print("=" * 60)
    
    # Load tokenizer
    print(f"📥 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_reviews['review_text'].tolist(),
        df_reviews['is_spoiler'].astype(int).tolist(),
        test_size=test_size,
        random_state=42,
        stratify=df_reviews['is_spoiler']
    )
    
    print(f"📊 Data split:")
    print(f"   Training: {len(train_texts)} samples")
    print(f"   Testing: {len(test_texts)} samples")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    # Create datasets
    print(f"🔤 Tokenizing with max_length={max_length}...")
    
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'labels': train_labels
    })
    
    test_dataset = Dataset.from_dict({
        'text': test_texts, 
        'labels': test_labels
    })
    
    # Apply tokenization
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    print(f"✅ Tokenization complete")
    print(f"   Train dataset: {len(train_dataset)} samples")
    print(f"   Test dataset: {len(test_dataset)} samples")
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'tokenizer': tokenizer,
        'train_texts': train_texts,
        'test_texts': test_texts,
        'train_labels': train_labels,
        'test_labels': test_labels
    }


def train_transformer_model(data_dict: Dict[str, Any],
                          model_name: str = "microsoft/deberta-v3-base",
                          config: Any = None,
                          num_epochs: int = 3,
                          learning_rate: float = 2e-5,
                          batch_size: int = 16) -> Dict[str, Any]:
    """
    Train a transformer model for spoiler detection.
    
    Args:
        data_dict: Dictionary with datasets from prepare_transformer_data
        model_name: Hugging Face model name
        config: Environment configuration
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Training batch size
        
    Returns:
        Dict with trained model and training results
    """
    print(f"\n🚀 TRAINING TRANSFORMER MODEL: {model_name}")
    print("=" * 60)
    
    # Load model
    print(f"📥 Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Move to device
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    model.to(device)
    
    # Training arguments
    output_dir = config.output_dir / f"transformer_{model_name.split('/')[-1]}" if config else "./transformer_output"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs") if config else "./logs",
        logging_steps=100,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to=None,  # Disable wandb/tensorboard
        dataloader_pin_memory=False  # Helps with some GPU issues
        # Note: MPS will be used automatically if available in newer Transformers
    )
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_dict['train_dataset'],
        eval_dataset=data_dict['test_dataset'],
        tokenizer=data_dict['tokenizer'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    print(f"🏋️ Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    
    training_result = trainer.train()
    
    print(f"✅ Training completed!")
    print(f"   Final train loss: {training_result.training_loss:.4f}")
    
    # Evaluate
    print(f"📊 Evaluating model...")
    eval_result = trainer.evaluate()
    
    print(f"✅ Evaluation completed!")
    for metric, value in eval_result.items():
        if not metric.startswith('eval_'):
            continue
        metric_name = metric.replace('eval_', '').capitalize()
        print(f"   {metric_name}: {value:.4f}")
    
    # Save model
    if config:
        model_path = config.output_dir / f"{model_name.split('/')[-1]}_model"
        trainer.save_model(str(model_path))
        print(f"💾 Model saved to: {model_path}")
    
    return {
        'model': model,
        'trainer': trainer,
        'tokenizer': data_dict['tokenizer'],
        'training_result': training_result,
        'eval_result': eval_result,
        'test_labels': data_dict['test_labels'],
        'model_name': model_name
    }


def evaluate_transformer_model(transformer_results: Dict[str, Any],
                             data_dict: Dict[str, Any],
                             config: Any = None) -> Dict[str, Any]:
    """
    Evaluate transformer model with comprehensive metrics.
    
    Args:
        transformer_results: Results from train_transformer_model
        data_dict: Data dictionary with test data
        config: Environment configuration
        
    Returns:
        Dict with comprehensive evaluation results
    """
    print(f"\n📊 COMPREHENSIVE TRANSFORMER EVALUATION")
    print("=" * 60)
    
    model = transformer_results['model']
    trainer = transformer_results['trainer']
    test_labels = transformer_results['test_labels']
    model_name = transformer_results['model_name']
    
    # Get predictions
    print("🔮 Generating predictions...")
    predictions = trainer.predict(data_dict['test_dataset'])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    
    # Use enhanced ModelEvaluator
    evaluator = ModelEvaluator(output_dir=str(config.output_dir) if config else "./outputs")
    
    # Evaluate with our enhanced metrics (includes specificity!)
    evaluation_results = evaluator.evaluate_model(
        model_name=f"Transformer_{model_name.split('/')[-1]}",
        y_true=np.array(test_labels),
        y_pred=y_pred,
        y_proba=y_proba,
        save_plots=True
    )
    
    print(f"✅ Comprehensive evaluation completed!")
    
    return {
        'evaluator': evaluator,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_true': test_labels,
        'evaluation_results': evaluation_results,
        'transformer_results': transformer_results
    }


def main():
    """Main execution function."""
    print("🎬 SPOILERSHIELD - MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # Configuration options for debugging
    QUICK_TEST = False  # Set to True for faster debugging (change to False for full run)
    SKIP_HYPERPARAMETER_SEARCH = True  # Skip hyperparameter search for faster testing
    
    # Transformer training option
    TRAIN_TRANSFORMERS = True  # Set to False to skip transformer training
    TRANSFORMER_MODEL = "roberta-base"  # Model to use for transformer training (very reliable)
    
    # Initialize configuration
    config = EnvConfig()
    print(f"Environment: {config.env}")
    print(f"Output directory: {config.output_dir}")
    
    if QUICK_TEST:
        print("⚡ QUICK TEST MODE - Using reduced datasets and simplified models")
    
    # Step 1: Load processed data
    processed_data, metadata = load_processed_data(config)
    
    # Step 2: Load raw data for baseline model
    df_reviews = load_raw_data_for_baseline(config)
    
    # For quick testing, sample the data
    if QUICK_TEST:
        print("⚡ Sampling data for quick test...")
        df_reviews = df_reviews.sample(n=min(1000, len(df_reviews)), random_state=42)
        print(f"   Using {len(df_reviews)} samples")
    
    # Step 3: Train baseline model
    baseline_results = train_baseline_model(df_reviews, config)
    
    # Step 4: Hyperparameter search (optional)
    if SKIP_HYPERPARAMETER_SEARCH:
        print("⚡ Skipping hyperparameter search...")
        hyperparameter_results = {
            'best_score': baseline_results['eval_results']['metrics']['f1'],
            'best_params': {'skipped': True},
            'results_path': 'skipped'
        }
    else:
        hyperparameter_results = hyperparameter_search_baseline(df_reviews, config)
    
    # Step 5: Train additional models
    additional_results = train_additional_models(df_reviews, config)
    
    # Step 6: Comprehensive evaluation
    evaluation_summary = comprehensive_model_evaluation(
        baseline_results, additional_results, config
    )
    
    # Step 7: Save all results
    save_training_results(
        baseline_results, hyperparameter_results, 
        additional_results, evaluation_summary, config
    )
    
    # Step 8: Train transformer models (if enabled)
    transformer_results = None
    if TRAIN_TRANSFORMERS:
        print("\n" + "=" * 60)
        print("🤖 STARTING TRANSFORMER TRAINING")
        print("=" * 60)
        
        # Check transformer environment
        if setup_transformer_environment():
            try:
                # Prepare data for transformer training
                transformer_data = prepare_transformer_data(
                    df_reviews, 
                    model_name=TRANSFORMER_MODEL,
                    max_length=512,
                    test_size=0.2
                )
                
                # Train transformer model
                transformer_training_results = train_transformer_model(
                    transformer_data,
                    model_name=TRANSFORMER_MODEL,
                    config=config,
                    num_epochs=3,
                    learning_rate=2e-5,
                    batch_size=16 if not QUICK_TEST else 8
                )
                
                # Comprehensive evaluation
                transformer_evaluation = evaluate_transformer_model(
                    transformer_training_results,
                    transformer_data,
                    config=config
                )
                
                transformer_results = {
                    'training': transformer_training_results,
                    'evaluation': transformer_evaluation,
                    'data': transformer_data
                }
                
                print("🎉 Transformer training completed successfully!")
                
            except Exception as e:
                print(f"❌ Transformer training failed: {e}")
                print("   Continuing with traditional model results...")
                transformer_results = None
        else:
            print("⚠️ Transformer environment not ready - skipping transformer training")
    else:
        print("\n🚫 Transformer training disabled")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING & EVALUATION COMPLETE!")
    print("=" * 60)
    
    # Show best traditional model
    print(f"\n📊 Traditional ML Best Model: {evaluation_summary['best_model']}")
    print(f"📊 Traditional ML Best F1 Score: {evaluation_summary['best_f1_score']:.4f}")
    
    # Show transformer results if available
    if transformer_results:
        transformer_eval = transformer_results['evaluation']['evaluation_results']
        transformer_name = f"Transformer_{TRANSFORMER_MODEL.split('/')[-1]}"
        print(f"\n🤖 Transformer Model: {transformer_name}")
        print(f"🤖 Transformer F1 Score: {transformer_eval['metrics']['f1']:.4f}")
        print(f"🤖 Transformer Accuracy: {transformer_eval['metrics']['accuracy']:.4f}")
        print(f"🤖 Transformer Specificity: {transformer_eval['metrics']['specificity']:.4f}")
        
        # Compare with best traditional model
        improvement = transformer_eval['metrics']['f1'] - evaluation_summary['best_f1_score']
        print(f"\n📈 F1 Improvement: {improvement:+.4f} ({improvement/evaluation_summary['best_f1_score']*100:+.1f}%)")
    
    print("\nNext steps:")
    print("1. Review model performance reports")
    print("2. Analyze hyperparameter search results") 
    if transformer_results:
        print("3. ✅ Transformer model trained successfully!")
        print("4. Compare transformer vs traditional model performance")
        print("5. Copy complete pipeline to notebook for Kaggle/Colab")
    else:
        print("3. Set TRAIN_TRANSFORMERS=True for transformer training")
        print("4. Copy working code to 02_model_training.ipynb")


if __name__ == "__main__":
    main() 