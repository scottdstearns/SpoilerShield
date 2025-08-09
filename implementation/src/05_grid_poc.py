#!/usr/bin/env python3
"""
PoC: Simplified Grid Search (Logistic Regression only) with explicit CV folds export.
- 3-fold Stratified CV
- Minimal parameter grid
- Exposes (param_combos, n_splits, total_evals) so GA can match budget
"""

from __future__ import annotations
import time
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, make_scorer

# Project-specific
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
        try:
            from transformers import set_seed as transformers_set_seed
            transformers_set_seed(seed)
        except ImportError:
            pass
    except ImportError:
        pass  # PyTorch not available
    
    print(f"🔒 All random seeds set to: {seed}")


@dataclass
class GridPoCResult:
    best_params: Dict[str, Any]
    best_cv_score: float
    test_metrics: Dict[str, float]
    optimization_time: float
    param_combos: int
    n_splits: int
    total_evals: int
    cv_indices: List[Tuple[np.ndarray, np.ndarray]]
    best_so_far_f1: List[float]
    best_so_far_auc: List[float]

class GridSearchPoC:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Set all seeds for reproducibility
        set_all_seeds(random_state)

    def prepare_data(self, df_reviews: pd.DataFrame):
        texts = df_reviews['review_text'].values
        labels = df_reviews['is_spoiler'].values
        return train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=self.random_state)

    def param_grid(self) -> Dict[str, List]:
        # Expanded grid for meaningful GA evolution (15 combos; 45 evals @ 3-fold)
        return {
            'tfidf__max_features': [10000, 20000, 40000],   # 3
            'classifier__C': [0.5, 1.0, 2.0, 5.0, 10.0],    # 5
            'classifier__penalty': ['l2'],                  # 1
        }  # => 15 combos

    def run(self, X_train, y_train, X_test, y_test) -> GridPoCResult:
        param_grid = self.param_grid()
        param_combos = 3 * 5 * 1
        n_splits = 3
        total_evals = param_combos * n_splits

        # Fixed folds (to share with GA)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_indices = list(skf.split(X_train, y_train))

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode')),
            ('classifier', LogisticRegression(random_state=self.random_state, solver='lbfgs', n_jobs=1, max_iter=2000, class_weight='balanced'))
        ])

        scorers = {'f1': make_scorer(f1_score), 'roc_auc': 'roc_auc'}

        start = time.time()
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_indices,  # use the exact same indices sequence
            scoring=scorers,
            refit='f1',
            n_jobs=-1,
            verbose=0,
            return_train_score=False,
        )
        gs.fit(X_train, y_train)
        elapsed = time.time() - start

        # candidate-wise means
        cvres = gs.cv_results_
        mean_f1 = cvres['mean_test_f1']
        mean_auc = cvres['mean_test_roc_auc']
        # Build best-so-far per evaluation assuming each candidate consumes all folds
        # Order is the order used by GridSearchCV
        best_so_far_f1 = []
        best_so_far_auc = []
        cur_best_f1 = -1.0
        cur_best_auc = -1.0
        for i in range(len(mean_f1)):
            cur_best_f1 = max(cur_best_f1, float(mean_f1[i]))
            cur_best_auc = max(cur_best_auc, float(mean_auc[i]))
            # Expand by n_splits to align with eval-count granularity
            for _ in range(n_splits):
                best_so_far_f1.append(cur_best_f1)
                best_so_far_auc.append(cur_best_auc)
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics = dict(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred)),
            recall=float(recall_score(y_test, y_pred)),
            f1=float(f1_score(y_test, y_pred)),
            roc_auc=float(roc_auc_score(y_test, y_proba)),
            specificity=float(tn / (tn + fp) if (tn + fp) else 0.0),
        )

        return GridPoCResult(
            best_params=gs.best_params_,
            best_cv_score=float(gs.best_score_),
            test_metrics=metrics,
            optimization_time=float(elapsed),
            param_combos=param_combos,
            n_splits=n_splits,
            total_evals=total_evals,
            cv_indices=cv_indices,
            best_so_far_f1=best_so_far_f1,
            best_so_far_auc=best_so_far_auc,
        )


def main():
    cfg = EnvConfig()
    dl = DataLoader(
        movie_reviews_path=cfg.get_data_path('train_reviews.json'),
        movie_details_path=cfg.get_data_path('IMDB_movie_details.json')
    )
    df = dl.load_imdb_movie_reviews()

    X_train, X_test, y_train, y_test = GridSearchPoC().prepare_data(df)
    res = GridSearchPoC().run(X_train, y_train, X_test, y_test)

    out = cfg.output_dir / "grid_poc_summary.json"
    out.write_text(__import__('json').dumps({
        "best_params": res.best_params,
        "best_cv_score": res.best_cv_score,
        "test_metrics": res.test_metrics,
        "optimization_time": res.optimization_time,
        "param_combos": res.param_combos,
        "n_splits": res.n_splits,
        "total_evals": res.total_evals,
    }, indent=2))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
