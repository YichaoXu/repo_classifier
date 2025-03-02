"""
README processing module.

This module provides functionality for fetching and analyzing README files
from GitHub repositories.
"""

from .heuristic import classify_description_heuristic
from .ai_classifier import classify_description_ai

__all__ = [
    'classify_description_heuristic',
    'classify_description_ai'
]
