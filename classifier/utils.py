"""
Utility functions for repository classification.

This module provides common utility functions used across the library.
"""

from typing import Dict

# Export functions
__all__ = [
    'normalize_scores',
    'get_top_n_scores'
]

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores to 0-1 range.
    
    Args:
        scores: Dictionary of scores
        
    Returns:
        Dictionary with normalized scores
    """
    if not scores:
        return {}
    
    max_score = max(scores.values()) if scores else 1
    if max_score == 0:
        return {k: 0.0 for k in scores}
    
    return {k: v / max_score for k, v in scores.items()}

def get_top_n_scores(scores: Dict[str, float], n: int) -> Dict[str, float]:
    """
    Get top N scores from a dictionary.
    
    Args:
        scores: Dictionary of scores
        n: Number of top scores to return
        
    Returns:
        Dictionary with top N scores
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:n])
