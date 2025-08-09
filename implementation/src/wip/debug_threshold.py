#!/usr/bin/env python3
"""
Quick debug script to understand threshold and prediction issues.
"""
import sys
from pathlib import Path
import numpy as np
import pickle

# Add src to path
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.env_config import EnvConfig

def analyze_model_predictions():
    """Analyze a trained model's predictions and probabilities."""
    
    config = EnvConfig()
    
    # Try to load a trained model (adjust path as needed)
    model_paths = [
        config.output_dir / 'baseline_model.pkl',
        config.output_dir / 'random_forest_model.pkl'
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            print(f"🔍 Analyzing {model_path.name}...")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract model and test data
            if 'model' in model_data:
                model = model_data['model']
                print(f"   Model type: {type(model).__name__}")
                
                # Check if we have TF-IDF data
                if 'tfidf' in model_data:
                    print("   ✅ Has TF-IDF vectorizer")
                    
                # Check model parameters
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    if 'class_weight' in params:
                        print(f"   Class weight: {params['class_weight']}")
                    if 'C' in params:
                        print(f"   Regularization C: {params['C']}")
                        
                print(f"   Metrics: {model_data.get('metrics', 'Not available')}")
                
            break
    else:
        print("❌ No trained models found. Run training first.")
        return
    
    # Check class distribution in raw data
    try:
        import pandas as pd
        data_path = config.data_dir / 'train_reviews.json'
        if data_path.exists():
            df = pd.read_json(data_path)
            print(f"\n📊 Raw data class distribution:")
            print(f"   Total samples: {len(df)}")
            spoiler_counts = df['is_spoiler'].value_counts()
            print(f"   Non-spoilers (False): {spoiler_counts.get(False, 0)} ({spoiler_counts.get(False, 0)/len(df)*100:.1f}%)")
            print(f"   Spoilers (True): {spoiler_counts.get(True, 0)} ({spoiler_counts.get(True, 0)/len(df)*100:.1f}%)")
            print(f"   Ratio (non-spoiler/spoiler): {spoiler_counts.get(False, 0) / spoiler_counts.get(True, 1):.2f}")
    except Exception as e:
        print(f"⚠️ Could not load raw data: {e}")

if __name__ == "__main__":
    analyze_model_predictions()