"""
README processing module.

This module provides functionality for fetching and analyzing README files
from GitHub repositories.
"""

from .fetcher import get_repo_readme
from .heuristic import heuristic_classify
from .ai_classifier import ai_classify

__all__ = [
    'get_repo_readme',
    'heuristic_classify',
    'ai_classifier'
]
