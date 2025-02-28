"""
Heuristic classification functionality.

This module provides keyword-based heuristic classification for repositories.
"""

from typing import Dict

from ..utils import normalize_scores

# Export functions
__all__ = ['heuristic_classify']

def heuristic_classify(readme_text: str, project_types: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """
    Classify repository based on keywords and weights.
    
    Args:
        readme_text: README content
        project_types: Project types and their keyword weight mappings
    
    Returns:
        Dictionary of project types and their normalized scores
    """
    # Preprocess README text (convert to lowercase)
    processed_text = readme_text.lower()
    
    # Calculate score for each project type
    scores = {}
    for project_type, keywords in project_types.items():
        type_score = 0
        for keyword, weight in keywords.items():
            # Simple calculation: keyword occurrences * weight
            occurrences = processed_text.count(keyword.lower())
            type_score += occurrences * weight
        
        scores[project_type] = type_score
    
    # Normalize scores to 0-1 range
    return normalize_scores(scores)
