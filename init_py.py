"""
Student Performance Prediction Project

This package contains modules for predicting student performance based on
demographic and educational factors.

Modules:
- data_preprocessing: Data cleaning and preprocessing utilities
- model_training: Machine learning model training and tuning
- evaluation: Model evaluation and performance metrics
- feature_engineering: Feature creation and selection utilities
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator

__all__ = [
    'DataPreprocessor',
    'ModelTrainer', 
    'ModelEvaluator'
]