"""
Utilities for Kaggle integration.

This module provides functions and classes to help manage the integration
between the local development environment and Kaggle notebooks.
"""

import os
import sys
from pathlib import Path
from typing import Optional 

def is_kaggle_environment() -> bool:
    """
    Check if the code is running in a Kaggle environment.
    
    Returns:
        bool: True if running in Kaggle, False otherwise
    """
    return (os.path.exists('/kaggle/input') and 
            len(os.listdir('/kaggle/input')) > 0)

def is_colab_environment() -> bool:
    """
    Check if the code is running in a Google Colab environment.
    
    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def detect_environment() -> str:
    """
    Detect the current environment.
    
    Returns:
        str: 'kaggle', 'colab', or 'local'
    """
    if is_kaggle_environment():
        return 'kaggle'
    elif is_colab_environment():
        return 'colab'
    else:
        return 'local'

def get_kaggle_data_path() -> Path:
    """
    Get the path to the Kaggle data directory.
    
    Returns:
        Path: Path to the Kaggle data directory
    """
    # Check if we're running in Kaggle
    if is_kaggle_environment():
        return Path('/kaggle/input')
    # Local development path
    return Path('implementation/data')

def get_kaggle_output_path() -> Path:
    """
    Get the path to the Kaggle output directory.
    
    Returns:
        Path: Path to the Kaggle output directory
    """
    # Check if we're running in Kaggle
    if os.path.exists('/kaggle/working'):
        return Path('/kaggle/working')
    # Local development path
    return Path('implementation/output')

def setup_kaggle_environment() -> None:
    """
    Set up the Kaggle environment.
    This includes:
    1. Installing required packages
    2. Setting up paths
    3. Downloading any required data
    
    Note: This should be called at the start of each Kaggle notebook.
    """
    if is_kaggle_environment():
        # Add any Kaggle-specific setup here
        pass
    else:
        print("Not running in Kaggle environment") 