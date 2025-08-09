#!/usr/bin/env python3
"""
PoC: A/B runner (Grid vs GA) – Logistic Regression only, apples-to-apples.
- Runs GridPoC then GAPoC in strict A/B mode (matched folds + eval budget)
- Optionally runs GA-advantage mode (continuous C) as a second experiment
- Writes a compact Markdown report
"""

import time, json, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.env_config import EnvConfig
from eda.data_loader import DataLoader

# Simple approach: run scripts directly rather than complex imports
import subprocess
import sys


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
    
    # Matplotlib for consistent plot generation
    plt.rcParams['figure.max_open_warning'] = 0
    
    print(f"🔒 All random seeds set to: {seed}")


def run():
    # Set global seed for the entire A/B test
    set_all_seeds(42)
    
    cfg = EnvConfig()
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🏃 Running GridSearch PoC...")
    result = subprocess.run([sys.executable, "05_grid_poc.py"], 
                          capture_output=True, text=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"❌ GridSearch failed: {result.stderr}")
        return
    
    print("🧬 Running GA PoC...")
    result = subprocess.run([sys.executable, "06_ga_poc.py"], 
                          capture_output=True, text=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"❌ GA failed: {result.stderr}")
        return

    # Load results from JSON files
    grid_json = cfg.output_dir / "grid_poc_summary.json"
    ga_json = cfg.output_dir / "ga_poc_summary.json"
    
    if not grid_json.exists() or not ga_json.exists():
        print(f"❌ Missing result files: {grid_json.exists()=}, {ga_json.exists()=}")
        return
    
    grid_res = json.loads(grid_json.read_text())
    ga_res = json.loads(ga_json.read_text())

    print(f"✅ GridSearch F1: {grid_res['test_metrics']['f1']:.4f}")
    print(f"✅ GA F1: {ga_res['test_f1']:.4f}")
    print(f"✅ GridSearch Time: {grid_res['optimization_time']:.1f}s")
    print(f"✅ GA Time: {ga_res['optimization_time']:.1f}s")

    # Generate simplified report
    md = out_dir / "ab_poc_report.md"
    def fmt(v): 
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    
    # Create simplified report with available data
    md.write_text(f"""# PoC A/B: GridSearch vs GA (LogReg only)

**Budget parity:** {grid_res['total_evals']} CV evaluations (3-fold × {grid_res['param_combos']} combos)  
**Same folds:** Yes (fixed StratifiedKFold indices)

## Results (held-out test F1)

- **GridSearch (strict grid)**: F1={fmt(grid_res['test_metrics']['f1'])}, ROC AUC={fmt(grid_res['test_metrics']['roc_auc'])}
- **GA (strict A/B, discrete grid)**: F1={fmt(ga_res['test_f1'])}, Time={fmt(ga_res['optimization_time'])}s

## Best Params

- **GridSearch**: `{grid_res['best_params']}`
- **GA (strict)**: `{ga_res['best_params']}`

## Performance Comparison

| Method | Test F1 | CV F1 | Time (s) | Evaluations |
|--------|---------|-------|----------|-------------|
| GridSearch | {fmt(grid_res['test_metrics']['f1'])} | {fmt(grid_res['best_cv_score'])} | {fmt(grid_res['optimization_time'])} | {grid_res['total_evals']} |
| GA (strict) | {fmt(ga_res['test_f1'])} | {fmt(ga_res['best_cv_score'])} | {fmt(ga_res['optimization_time'])} | {ga_res['total_evals']} |

## Key Findings

**Winner by F1 Score**: {"GridSearch" if grid_res['test_metrics']['f1'] > ga_res['test_f1'] else "GA"}
**Efficiency Winner**: {"GridSearch" if grid_res['optimization_time'] < ga_res['optimization_time'] else "GA"}

## Notes
- GA uses tournament selection + uniform crossover + adaptive mutation
- Budget parity ensures fair comparison (same CV folds, same evaluation count)
- This is a proof-of-concept for larger-scale A/B testing

""")
    print(f"📋 Report: {md}")

if __name__ == "__main__":
    run()
