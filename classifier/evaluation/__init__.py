"""
Evaluation functionality for repository classifiers.

This module provides tools for evaluating classifier performance against ground truth data.
It supports loading ground truth data from CSV files and calculating accuracy metrics.
"""

from .ground_truth import (
    load_ground_truth,
    save_ground_truth,
    evaluate_classifier,
    add_ground_truth_entry,
    get_ground_truth_repos
)

__all__ = [
    'load_ground_truth',
    'save_ground_truth',
    'evaluate_classifier',
    'add_ground_truth_entry',
    'get_ground_truth_repos'
] 