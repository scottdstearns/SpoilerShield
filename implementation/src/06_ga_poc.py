#!/usr/bin/env python3
"""
PoC: Genetic Algorithm (Logistic Regression only)
- Reuses exact CV folds from GridSearchPoC
- Budget parity via max_evals
- Modes:
  * strict_ab=True  -> discrete grid identical to GridSearch space
  * strict_ab=False -> GA-advantage (continuous C in [1,5])
- Optional early stop (disabled by default)
"""

from __future__ import annotations
import time, copy, math, random, csv, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, make_scorer

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
class GAResult:
    best_params: Dict[str, Any]
    best_cv_score: float
    test_f1: float
    test_auc: float
    optimization_time: float
    total_evals: int
    best_so_far_f1: List[float]
    best_so_far_auc: List[float]

class GAPoC:
    def __init__(
        self,
        random_state: int = 42,
        population_size: int = 5,
        generations: int = 3,  # With 3-fold CV and 15-combo grid -> 45 evals parity
        tournament_k: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        strict_ab: bool = True,
        max_evals: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
        early_stop_epsilon: float = 1e-4,
        log_gen_stats: bool = False,
        run_label: str = 'strict',
    ):
        self.rs = np.random.RandomState(random_state)
        self.random_state = random_state
        # Set all seeds for reproducibility
        set_all_seeds(random_state)
        self.population_size = population_size
        self.generations = generations
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.strict_ab = strict_ab
        self.max_evals = max_evals
        self.early_stop_patience = early_stop_patience
        self.early_stop_epsilon = early_stop_epsilon
        self.log_gen_stats = log_gen_stats
        self.run_label = run_label
        self.log_gen_stats = log_gen_stats

        self._cv_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self._eval_count = 0
        self._best_f1_history: List[float] = []
        self._best_auc_history: List[float] = []

    # --- data & folds ---
    def prepare_data(self, df_reviews: pd.DataFrame):
        texts = df_reviews['review_text'].values
        labels = df_reviews['is_spoiler'].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=self.random_state
        )
        # default CV (3-fold) – can be overridden by set_cv_indices()
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        self._cv_indices = list(skf.split(X_tr, y_tr))
        return X_tr, X_te, y_tr, y_te

    def set_cv_indices(self, cv_indices: List[Tuple[np.ndarray, np.ndarray]]):
        self._cv_indices = cv_indices

    # --- search space ---
    def search_space(self):
        if self.strict_ab:
            # Match expanded grid from GridSearchPoC: 3 × 5 × 1 = 15 combinations
            return {
                'tfidf__max_features': {'type': 'discrete', 'values': [10000, 20000, 40000]},
                'classifier__C': {'type': 'discrete', 'values': [0.5, 1.0, 2.0, 5.0, 10.0]},
                'classifier__penalty': {'type': 'categorical', 'values': ['l2']},
            }
        else:
            # GA advantage: continuous C parameter
            return {
                'tfidf__max_features': {'type': 'discrete', 'values': [10000, 20000, 40000]},
                'classifier__C': {'type': 'continuous', 'min': 0.5, 'max': 10.0},
                'classifier__penalty': {'type': 'categorical', 'values': ['l2']},
            }

    # --- GA operators ---
    def create_individual(self, space: Dict[str, Dict]) -> List[Any]:
        genes = []
        for name, cfg in space.items():
            t = cfg['type']
            if t == 'discrete':
                genes.append(self.rs.choice(cfg['values']))
            elif t == 'categorical':
                genes.append(self.rs.randint(0, len(cfg['values'])))  # index
            else:  # continuous
                genes.append(self.rs.uniform(cfg['min'], cfg['max']))
        return genes

    def decode(self, indiv: List[Any], space: Dict[str, Dict]) -> Dict[str, Any]:
        params = {}
        i = 0
        for name, cfg in space.items():
            t = cfg['type']
            if t == 'discrete':
                params[name] = int(indiv[i])
            elif t == 'categorical':
                idx = int(round(indiv[i])) % len(cfg['values'])
                params[name] = cfg['values'][idx]
            else:
                # continuous
                val = float(indiv[i])
                val = max(cfg['min'], min(cfg['max'], val))
                params[name] = val
            i += 1
        return params

    def tournament_select(self, pop: List[List[Any]], fits: List[float]) -> List[Any]:
        idxs = self.rs.choice(len(pop), size=min(self.tournament_k, len(pop)), replace=False)
        best = max(idxs, key=lambda j: fits[j])
        return pop[best]

    def crossover(self, p1: List[Any], p2: List[Any]) -> Tuple[List[Any], List[Any]]:
        if self.rs.rand() > self.crossover_rate:
            return copy.deepcopy(p1), copy.deepcopy(p2)
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        for i in range(len(c1)):
            if self.rs.rand() < 0.5:
                c1[i], c2[i] = c2[i], c1[i]
        return c1, c2

    def mutate(self, indiv: List[Any], space: Dict[str, Dict], gen: int) -> List[Any]:
        out = copy.deepcopy(indiv)
        rate = self.mutation_rate * (1.0 - gen / max(1, self.generations))
        for i, (name, cfg) in enumerate(space.items()):
            if self.rs.rand() < rate:
                if cfg['type'] == 'continuous':
                    span = (cfg['max'] - cfg['min'])
                    out[i] = float(np.clip(out[i] + self.rs.normal(0, 0.1*span), cfg['min'], cfg['max']))
                elif cfg['type'] == 'discrete':
                    vals = list(cfg['values'])
                    cand = [v for v in vals if v != out[i]] or vals
                    out[i] = int(self.rs.choice(cand))
                else:  # categorical
                    k = len(cfg['values'])
                    out[i] = int(self.rs.randint(0, k))
        return out

    # --- fitness ---
    def _cv_metrics(self, params: Dict[str, Any], X_train, y_train) -> tuple:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', lowercase=True, strip_accents='unicode',
                max_features=params['tfidf__max_features'],
            )),
            ('classifier', LogisticRegression(
                random_state=self.random_state, solver='lbfgs', n_jobs=1, max_iter=2000,
                penalty=params['classifier__penalty'], C=params['classifier__C'],
                class_weight='balanced'
            ))
        ])
        # Full-CV-or-stop: if not enough budget for a complete CV evaluation, abort evaluation of this individual.
        if self.max_evals is not None:
            remaining = self.max_evals - self._eval_count
            if remaining < len(self._cv_indices):
                # Signal to caller by returning NaN; caller will stop the run.
                return float('nan')
        f1s = []
        aucs = []
        for tr_idx, va_idx in self._cv_indices:
            pipeline.fit(X_train[tr_idx], y_train[tr_idx])
            y_pred = pipeline.predict(X_train[va_idx])
            y_proba = pipeline.predict_proba(X_train[va_idx])[:, 1]
            f1s.append(f1_score(y_train[va_idx], y_pred))
            try:
                aucs.append(roc_auc_score(y_train[va_idx], y_proba))
            except ValueError:
                # Single-class edge case in a fold
                pass
            self._eval_count += 1
        f1_mean = float(np.mean(f1s)) if f1s else 0.0
        auc_mean = float(np.mean(aucs)) if aucs else 0.0
        return f1_mean, auc_mean

    # --- main optimize ---
    def optimize(self, X_train, y_train, X_test, y_test) -> GAResult:
        space = self.search_space()
        pop = [self.create_individual(space) for _ in range(self.population_size)]
        best_fit = -1.0
        best_indiv = None
        last_improve_gen = 0
        gen_rows = []  # For generation statistics tracking

        start = time.time()
        # Optional per-generation stats log
        if self.log_gen_stats:
            stats_path = Path(EnvConfig().output_dir) / 'ga_gen_stats.csv'
            with open(stats_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['generation','best_f1','mean_f1','std_f1','best_auc','mean_auc','std_auc','unique_configs'])
        for gen in range(self.generations):
            # Evaluate
            fits = []
            aucs = []
            stop_now = False
            for indiv in pop:
                params = self.decode(indiv, space)
                f1_mean, auc_mean = self._cv_metrics(params, X_train, y_train)
                if isinstance(f1_mean, float) and math.isnan(f1_mean):
                    stop_now = True
                    break
                fits.append(f1_mean)
                aucs.append(auc_mean)
            if stop_now:
                print('⏹️ Stopping before evaluating incomplete individual due to budget parity.')
                break
            # Log per-generation stats if enabled
            if self.log_gen_stats and fits:
                unique_configs = len({tuple(self.decode(ind, space).items()) for ind in pop})
                with open(stats_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([gen, max(fits), float(np.mean(fits)), float(np.std(fits)),
                                     max(aucs), float(np.mean(aucs)), float(np.std(aucs)), unique_configs])
            # Track best and gen stats
            gen_best_f1 = max(fits) if fits else -1.0
            gen_best_auc = max(aucs) if aucs else -1.0
            gen_mean_f1 = float(np.mean(fits)) if fits else -1.0
            gen_std_f1 = float(np.std(fits)) if fits else -1.0
            gen_mean_auc = float(np.mean(aucs)) if aucs else -1.0
            gen_std_auc = float(np.std(aucs)) if aucs else -1.0
            # diversity proxy: unique decoded configs in this generation
            unique_configs = 0
            try:
                decoded = [tuple(sorted(self.decode(ind, space).items())) for ind in pop]
                unique_configs = len(set(decoded))
            except Exception:
                pass
            gen_rows.append({
                'gen': gen,
                'best_f1': float(gen_best_f1),
                'mean_f1': gen_mean_f1,
                'std_f1': gen_std_f1,
                'best_auc': float(gen_best_auc),
                'mean_auc': gen_mean_auc,
                'std_auc': gen_std_auc,
                'unique_configs': int(unique_configs),
                'evals_so_far': int(self._eval_count)
            })
            # Per-eval history: expand by CV folds for each individual to align with eval count
            # We approximate by logging gen-level best per eval within the gen (population * folds)
            per_gen_evals = len(self._cv_indices) * len(pop)
            if not self._best_f1_history:
                cur_f1 = -1.0
                cur_auc = -1.0
            else:
                cur_f1 = self._best_f1_history[-1]
                cur_auc = self._best_auc_history[-1]
            cur_f1 = max(cur_f1, gen_best_f1)
            cur_auc = max(cur_auc, gen_best_auc)
            for _ in range(per_gen_evals):
                self._best_f1_history.append(cur_f1)
                self._best_auc_history.append(cur_auc)
            if gen_best_f1 > best_fit + (self.early_stop_epsilon or 0.0):
                best_fit = gen_best_f1
                best_indiv = copy.deepcopy(pop[int(np.argmax(fits))])
                last_improve_gen = gen
            # Budget stop
            if self.max_evals is not None and self._eval_count >= self.max_evals:
                print(f"⏹️ Stopping at gen {gen}: reached eval budget {self._eval_count}/{self.max_evals}")
                break
            # Early stop
            if self.early_stop_patience is not None and (gen - last_improve_gen) >= self.early_stop_patience:
                print(f"⏹️ Early stop: no improvement for {self.early_stop_patience} generations")
                break
            # Selection
            selected = [self.tournament_select(pop, fits) for _ in range(self.population_size)]
            # Next gen
            next_pop = []
            for i in range(0, self.population_size, 2):
                p1 = selected[i]
                p2 = selected[(i+1) % self.population_size]
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1, space, gen)
                c2 = self.mutate(c2, space, gen)
                next_pop.extend([c1, c2])
            pop = next_pop[:self.population_size]

        # Final test on held-out set
        best_params = self.decode(best_indiv, space) if best_indiv is not None else self.decode(pop[0], space)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', lowercase=True, strip_accents='unicode',
                max_features=best_params['tfidf__max_features'],
            )),
            ('classifier', LogisticRegression(
                random_state=self.random_state, solver='lbfgs', n_jobs=1, max_iter=2000,
                penalty=best_params['classifier__penalty'], C=best_params['classifier__C'],
                class_weight='balanced'
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_f1 = float(f1_score(y_test, y_pred))
        try:
            test_auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            test_auc = 0.0

        # Write per-generation stats if enabled
        if self.log_gen_stats:
            try:
                cfg = EnvConfig()
                out_dir = cfg.output_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                import pandas as pd
                df = pd.DataFrame(gen_rows)
                csv_path = out_dir / f'ga_gen_stats_{self.run_label}.csv'
                df.to_csv(csv_path, index=False)
                print(f'Per-generation stats saved to: {csv_path}')
            except Exception as e:
                print(f'WARN: failed to write gen stats: {e}')

        return GAResult(
            best_params=best_params,
            best_cv_score=float(best_fit),
            test_f1=test_f1,
            test_auc=test_auc,
            optimization_time=float(time.time() - start),
            total_evals=int(self._eval_count),
            best_so_far_f1=list(self._best_f1_history),
            best_so_far_auc=list(self._best_auc_history),
        )

def main():
    cfg = EnvConfig()
    dl = DataLoader(
        movie_reviews_path=cfg.get_data_path('train_reviews.json'),
        movie_details_path=cfg.get_data_path('IMDB_movie_details.json')
    )
    df = dl.load_imdb_movie_reviews()

    ga = GAPoC(strict_ab=True)  # default strict A/B
    X_tr, X_te, y_tr, y_te = ga.prepare_data(df)

    # If a GridPoC JSON exists, use its folds & budget
    grid_json = cfg.output_dir / "grid_poc_summary.json"
    if grid_json.exists():
        data = __import__('json').loads(grid_json.read_text())
        # rebuild CV indices by re-splitting in the same way; or simply trust parity by same seed
        # Since Grid wrote exact indices only in-memory, we'll deterministically regenerate them here.
        skf = StratifiedKFold(n_splits=data["n_splits"], shuffle=True, random_state=42)
        ga.set_cv_indices(list(skf.split(X_tr, y_tr)))
        ga.max_evals = int(data["total_evals"])
        print(f"Matching budget: max_evals={ga.max_evals}")
    res = ga.optimize(X_tr, y_tr, X_te, y_te)

    out = cfg.output_dir / "ga_poc_summary.json"
    out.write_text(__import__('json').dumps({
        "best_params": res.best_params,
        "best_cv_score": res.best_cv_score,
        "test_f1": res.test_f1,
        "optimization_time": res.optimization_time,
        "total_evals": res.total_evals,
    }, indent=2))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
