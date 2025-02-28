"""
Repository Classifier Library - Analyze and classify GitHub repositories

This library provides two classification methods:
1. Keyword-based heuristic classification
2. AI-based classification

Users can use built-in classifier configurations or register custom ones.
"""

# Export core functionality
from .core import (
    classify_repository_heuristic,
    classify_repository_ai
)

# Export README processing functionality
from .readme import (
    get_repo_readme,
    heuristic_classify,
    ai_classify
)

# Export utility functions
from .utils import (
    normalize_scores,
    get_top_n_scores
)

# Export registry functionality
from .registry import (
    register_classifier,
    unregister_classifier,
    get_classifier,
    get_available_classifiers,
    load_classifier_from_module,
    create_classifier_from_file
)

# Export built-in configurations
from .predefine import (
    PHP_PROJECT_TYPES,
    PYTHON_PROJECT_TYPES,
    ALL_PROJECT_TYPES,
    ALL_PROJECT_TYPE_NAMES,
    CLASSIFIER_NAMES
)

# Version information
__version__ = '0.1.0'

# Define what's available in the public API
__all__ = [
    # Core functionality
    'classify_repository_heuristic',
    'classify_repository_ai',
    
    # README processing functionality
    'get_repo_readme',
    'heuristic_classify',
    'ai_classify',
    
    # Utility functions
    'normalize_scores',
    'get_top_n_scores',
    
    # Registry functionality
    'register_classifier',
    'unregister_classifier',
    'get_classifier',
    'get_available_classifiers',
    'load_classifier_from_module',
    'create_classifier_from_file',
    
    # Built-in configurations
    'PHP_PROJECT_TYPES',
    'PYTHON_PROJECT_TYPES',
    'ALL_PROJECT_TYPES',
    'ALL_PROJECT_TYPE_NAMES',
    'CLASSIFIER_NAMES',
    
    # Version
    '__version__'
]
