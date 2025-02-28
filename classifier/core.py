"""
Core functionality for repository classification.

This module provides the main API for classifying repositories.
"""

from typing import Dict, List, Optional, Union

# Import from other modules
from .registry import get_classifier, get_available_classifiers
from .utils import normalize_scores, get_top_n_scores
from .readme import get_repo_readme, heuristic_classify, ai_classify
from .predefine import CLASSIFIER_NAMES, ALL_PROJECT_TYPE_NAMES

# Export functions
__all__ = [
    'classify_repository_heuristic',
    'classify_repository_ai'
]

def classify_repository_heuristic(
    repo_url: str,
    classifier: Optional[Union[str, Dict[str, Dict[str, int]]]] = CLASSIFIER_NAMES.PHP,
    top_n: int = 3
) -> Dict[str, float]:
    """
    Classify a GitHub repository using keyword-based heuristic method.
    
    Args:
        repo_url: GitHub repository URL
        classifier: Classifier name (str) or configuration dictionary (Dict)
            - If string, looks up the classifier in the registry
            - If dictionary, uses it directly as configuration
        top_n: Number of top types to return
        
    Returns:
        {"Type1": score, "Type2": score, "Type3": score}
        Scores are float values between 0-1
        
    Raises:
        ValueError: If repository URL is invalid, README cannot be found, or classifier is not found
    """
    # Parse classifier parameter
    if isinstance(classifier, str):
        # Get classifier from registry by name
        config = get_classifier(classifier)
        if not config:
            available = get_available_classifiers()
            raise ValueError(f"Classifier not found: {classifier}. Available classifiers: {', '.join(available)}")
    else:
        # Use dictionary directly as configuration
        config = classifier
    
    # Get README
    readme_text = get_repo_readme(repo_url)
    
    # Perform heuristic classification
    all_scores = heuristic_classify(readme_text, config)
    
    # Return top N scores
    return get_top_n_scores(all_scores, top_n)

def classify_repository_ai(
    repo_url: str,
    api_key: str,
    classifier: Optional[Union[str, List[str]]] = CLASSIFIER_NAMES.PHP,
    api_url: Optional[str] = None,
    model_name: str = "gpt-3.5-turbo",
    top_n: int = 3
) -> Dict[str, float]:
    """
    Classify a GitHub repository using AI service.
    
    Args:
        repo_url: GitHub repository URL
        api_key: API key for AI service
        classifier: Classifier name (str) or list of project types (List[str])
            - If string, looks up project types in the registry
            - If list, uses it directly as project types
        api_url: API URL for AI service
        model_name: AI model name
        top_n: Number of top types to return
        
    Returns:
        {"Type1": score, "Type2": score, "Type3": score}
        Scores are float values between 0-1
        
    Raises:
        ValueError: If repository URL is invalid, README cannot be found, classifier is not found,
                   or AI service returns an error
    """
    # Determine project types to use
    if isinstance(classifier, str):
        # Get project types from registry
        if classifier.lower() in ALL_PROJECT_TYPE_NAMES:
            project_types = ALL_PROJECT_TYPE_NAMES[classifier.lower()]
        else:
            available = get_available_classifiers()
            raise ValueError(f"Classifier not found: {classifier}. Available classifiers: {', '.join(available)}")
    else:
        # Use list directly as project types
        project_types = classifier
    
    # Get README
    readme_text = get_repo_readme(repo_url)
    
    # Perform AI classification
    all_scores = ai_classify(
        readme_text, 
        repo_url, 
        api_key, 
        api_url, 
        model_name,
        project_types
    )
    
    # Return top N scores
    return get_top_n_scores(all_scores, top_n) 