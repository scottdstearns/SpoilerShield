"""
Environment configuration for SpoilerShield project.

This module provides a centralized configuration class that handles
path management across different environments (Kaggle, Colab, Local).
"""

import os
import sys
from pathlib import Path
from typing import Optional
from .kaggle_utils import detect_environment, is_kaggle_environment, is_colab_environment


class EnvConfig:
    """
    Environment configuration class for managing paths across different environments.
    
    This class automatically detects the current environment and provides
    appropriate paths for data, code, and outputs.
    """
    
    def __init__(self, custom_root: Optional[str] = None, custom_output_dir: Optional[str] = None):
        """
        Initialize the environment configuration.
        
        Args:
            custom_root: Optional custom root directory path
            custom_output_dir: Optional custom output directory path
        """
        self._env = detect_environment()
        self._custom_root = custom_root
        self._custom_output_dir = custom_output_dir
        
        # Set up paths based on environment
        self._setup_paths()
    
    @property
    def env(self) -> str:
        """Get the current environment."""
        return self._env
    
    @property
    def root_dir(self) -> Path:
        """Get the project root directory."""
        return self._root_dir
    
    @property
    def src_dir(self) -> Path:
        """Get the source code directory."""
        return self._src_dir
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self._data_dir
    
    @property
    def output_dir(self) -> Path:
        """Get the output directory."""
        return self._output_dir
    
    @property
    def utils_dir(self) -> Path:
        """Get the utils directory."""
        return self._utils_dir
    
    @property
    def eda_dir(self) -> Path:
        """Get the EDA directory."""
        return self._eda_dir
    
    def _setup_paths(self):
        """Set up paths based on the detected environment."""
        if self._custom_root:
            self._root_dir = Path(self._custom_root)
        elif self._env == 'kaggle':
            self._root_dir = Path('/kaggle/input/spoiler-shield-code/spoiler_shield_code')
        elif self._env == 'colab':
            self._root_dir = Path('/content/drive/MyDrive/IK Course Materials/SpoilerShield')
        else:
            # Local environment - find project root intelligently
            current_path = Path.cwd()
            # If we're in implementation/src, go up to project root
            if current_path.name == 'src' and current_path.parent.name == 'implementation':
                self._root_dir = current_path.parent.parent
            # If we're in implementation, go up one level
            elif current_path.name == 'implementation':
                self._root_dir = current_path.parent
            # Otherwise assume we're in project root
            else:
                self._root_dir = current_path
        
        # Set up relative paths (same across all environments)
        self._src_dir = self._root_dir / 'implementation' / 'src'
        self._data_dir = self._src_dir / 'data'
        self._utils_dir = self._src_dir / 'utils'
        self._eda_dir = self._src_dir / 'eda'
        
        # Set up output directory
        if self._custom_output_dir:
            self._output_dir = Path(self._custom_output_dir)
        elif self._env == 'kaggle':
            self._output_dir = Path('/kaggle/working')
        elif self._env == 'colab':
            self._output_dir = self._root_dir / 'outputs'
        else:
            self._output_dir = self._root_dir / 'outputs'
        
        # Create output directory if it doesn't exist
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self, filename: str) -> Path:
        """
        Get the full path to a data file.
        
        Args:
            filename: Name of the data file
            
        Returns:
            Path: Full path to the data file
        """
        return self._data_dir / filename
    
    def get_output_path(self, filename: str) -> Path:
        """
        Get the full path for an output file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path: Full path for the output file
        """
        return self._output_dir / filename
    
    def get_src_path(self, *path_parts: str) -> Path:
        """
        Get a path within the src directory.
        
        Args:
            *path_parts: Path components relative to src directory
            
        Returns:
            Path: Full path within src directory
        """
        return self._src_dir.joinpath(*path_parts)
    
    def add_src_to_syspath(self):
        """Add the src directory to sys.path for imports."""
        src_path_str = str(self._src_dir)
        if src_path_str not in sys.path:
            sys.path.insert(0, src_path_str)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"EnvConfig(env={self._env}, root={self._root_dir})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return (f"EnvConfig(env='{self._env}', "
                f"root_dir={self._root_dir}, "
                f"src_dir={self._src_dir}, "
                f"data_dir={self._data_dir}, "
                f"output_dir={self._output_dir})")


# Convenience function to create a default config
def get_config(custom_root: Optional[str] = None, custom_output_dir: Optional[str] = None) -> EnvConfig:
    """
    Get a default environment configuration.
    
    Args:
        custom_root: Optional custom root directory path
        custom_output_dir: Optional custom output directory path
        
    Returns:
        EnvConfig: Configured environment configuration
    """
    return EnvConfig(custom_root=custom_root, custom_output_dir=custom_output_dir) 