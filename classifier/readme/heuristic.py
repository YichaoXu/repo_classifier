"""
Heuristic-based classification functionality.

This module provides keyword-based classification for repositories using a heuristic approach.
It analyzes README content to determine the most likely project types based on keyword frequency
and predefined weights.
"""

from typing import Dict, Union
from ..utils import normalize_scores

# Export functions
__all__ = ['classify_readme_heuristic']

def classify_readme_heuristic(
    readme_text: str, 
    classifier: Union[str, Dict[str, Dict[str, int]]]
) -> Dict[str, float]:
    """
    Classify repository README content based on keywords and weights using a heuristic approach.
    
    This function analyzes the README text content and calculates scores for different project types
    based on the occurrence of keywords and their assigned weights. It's a simple but effective
    method for determining the most likely project type without using AI.
    
    Args:
        readme_text: The README content to analyze.
                    This is a required parameter containing the raw text of the README file.
                    The text will be preprocessed (converted to lowercase) before analysis.
        
        classifier: Dictionary mapping project types to their keyword weight mappings.
                      This is a required parameter that defines the classification criteria.
                      Format:
                      {
                          "Project Type 1": {"keyword1": weight1, "keyword2": weight2, ...},
                          "Project Type 2": {"keyword3": weight3, "keyword4": weight4, ...},
                          ...
                      }
                      Example:
                      {
                          "Web Framework": {"laravel": 10, "routing": 5, "mvc": 5},
                          "CMS": {"content": 8, "management": 5, "admin": 5}
                      }
    
    Returns:
        A dictionary mapping project types to normalized confidence scores (0.0 to 1.0).
        The scores are normalized so that they sum to 1.0, making them comparable.
        Example: {"Web Framework": 0.75, "CMS": 0.25}
        
    Note:
        - The scoring is based on simple keyword counting and weighting
        - Higher weights increase the importance of specific keywords
        - The scores are normalized to make them comparable across different project types
        - This function does not filter or rank the results; use get_top_n_scores for that
    
    Examples:
        >>> # Classify README content with PHP project types
        >>> php_types = {
        ...     "Web Framework": {"laravel": 10, "routing": 5, "mvc": 5},
        ...     "CMS": {"content": 8, "management": 5, "admin": 5}
        ... }
        >>> classify_readme_heuristic(
        ...     "This is a Laravel web framework with routing capabilities.",
        ...     php_types
        ... )
        {'Web Framework': 0.8, 'CMS': 0.2}
    """
    # Preprocess README text (convert to lowercase for case-insensitive matching)
    processed_text = readme_text.lower()
    
    # Calculate score for each project type based on keyword occurrences
    scores = {}
    for project_type, keywords in classifier.items():
        # Initialize score for this project type
        type_score = 0
        
        # Iterate through each keyword and its weight
        for keyword, weight in keywords.items():
            # Count occurrences of the keyword in the processed text
            occurrences = processed_text.count(keyword.lower())
            
            # Add to the score: keyword occurrences * assigned weight
            type_score += occurrences * weight
        
        # Store the calculated score for this project type
        scores[project_type] = type_score
    
    # Normalize scores to 0-1 range for better comparability
    return normalize_scores(scores)
