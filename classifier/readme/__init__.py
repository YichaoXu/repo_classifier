"""
README processing module.

This module provides functionality for fetching and analyzing README files
from GitHub repositories.
"""

from .fetcher import get_repo_readme
from .heuristic import classify_readme_heuristic
from .ai_classifier import classify_readme_ai

__all__ = [
    'get_repo_readme',
    'classify_readme_heuristic',
    'ai_classifier'
]
