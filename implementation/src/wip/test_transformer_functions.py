#!/usr/bin/env python3
"""
Test script to verify transformer functions work correctly.
This simulates what would happen in a notebook environment.
"""
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Test import
try:
    from importlib import import_module
    training_module = import_module('02_model_training')
    
    # Check if functions exist
    functions = [
        'setup_transformer_environment',
        'prepare_transformer_data', 
        'train_transformer_model',
        'evaluate_transformer_model'
    ]
    
    for func_name in functions:
        if hasattr(training_module, func_name):
            print(f"✅ {func_name} available")
        else:
            print(f"❌ {func_name} missing")
            
except ImportError as e:
    print(f"❌ Import error: {e}")

# Test environment setup
def test_environment():
    """Test transformer environment setup."""
    print("\n" + "="*50)
    print("Testing Transformer Environment")
    print("="*50)
    
    try:
        # Get the function from the module
        setup_func = getattr(training_module, 'setup_transformer_environment')
        env_ready = setup_func()
        return env_ready
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Transformer Functions")
    test_environment()
    print("\n✅ Test completed!")