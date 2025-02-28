"""
Project type classifiers module.

This module exports all built-in project type configurations.
"""

# Import all configurations directly (no try/except)
from .php import PHP_PROJECT_TYPES, PHP_PROJECT_TYPE_NAMES
from .python import PYTHON_PROJECT_TYPES, PYTHON_PROJECT_TYPE_NAMES
from .javascript import JAVASCRIPT_PROJECT_TYPES, JAVASCRIPT_PROJECT_TYPE_NAMES

# Merge all configurations to create a complete mapping
ALL_PROJECT_TYPES = {
    "php": PHP_PROJECT_TYPES,
    "python": PYTHON_PROJECT_TYPES,
    "javascript": JAVASCRIPT_PROJECT_TYPES
}

# Export all project type names
DFT_PROJECT_TYPE_NAMES = {
    "php": PHP_PROJECT_TYPE_NAMES,
    "python": PYTHON_PROJECT_TYPE_NAMES,
    "javascript": JAVASCRIPT_PROJECT_TYPE_NAMES
}

# Define a constant object for classifier names
class CLASSIFIER_NAMES:
    """Constant object containing all default classifier names."""
    PHP = "php"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    
    @classmethod
    def all(cls) -> list:
        """Return a list of all available default classifier names."""
        return [cls.PHP, cls.PYTHON, cls.JAVASCRIPT]
    
    @classmethod
    def available(cls) -> list:
        """Return a list of actually available classifier names."""
        return list(ALL_PROJECT_TYPES.keys())