"""
Repository Classifier Library.

This library provides tools for classifying GitHub repositories based on their README content.
It supports both heuristic (keyword-based) and AI-powered classification methods.
"""

# Version
__version__ = '0.1.0'

# Core functionality
from .core import (
    classify_repository_heuristic,
    classify_repository_ai
)

# README processing functionality
from .readme import (
    get_repo_readme
)

# Utility functions
from .utils import (
    normalize_scores,
    get_top_n_scores
)

# Registry functionality
from .registry import (
    register_classifier,
    unregister_classifier,
    get_classifier,
    get_available_classifiers,
    load_classifier_from_module,
    create_classifier_from_file
)

# Evaluation functionality
from .evaluation import (
    load_ground_truth,
    save_ground_truth,
    evaluate_classifier,
    add_ground_truth_entry,
    get_ground_truth_repos
)

# Built-in configurations
from .predefine import (
    CLASSIFIER_NAMES,
    ALL_PROJECT_TYPES,
    DFT_PROJECT_TYPE_NAMES
)

# Define public API
__all__ = [
    # Version
    '__version__',
    
    # Core functionality
    'classify_repository_heuristic',
    'classify_repository_ai',
    
    # README processing
    'get_repo_readme',
    
    # Utilities
    'normalize_scores',
    'get_top_n_scores',
    
    # Registry
    'register_classifier',
    'unregister_classifier',
    'get_classifier',
    'get_available_classifiers',
    'load_classifier_from_module',
    'create_classifier_from_file',
    
    # Evaluation
    'load_ground_truth',
    'save_ground_truth',
    'evaluate_classifier',
    'add_ground_truth_entry',
    'get_ground_truth_repos',
    
    # Built-in configurations
    'CLASSIFIER_NAMES',
    'ALL_PROJECT_TYPES',
    'DFT_PROJECT_TYPE_NAMES'
]
