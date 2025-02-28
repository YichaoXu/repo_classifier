"""
Core functionality for repository classification.

This module provides the main API for classifying repositories.
"""

from typing import Dict, List, Optional, Union

# Import from other modules
from .registry import get_classifier, get_available_classifiers
from .utils import get_top_n_scores
from .readme import get_repo_readme, classify_readme_heuristic, classify_readme_ai
from .predefine import CLASSIFIER_NAMES, ALL_PROJECT_TYPE_NAMES

# Export functions
__all__ = [
    'classify_repository_heuristic',
    'classify_repository_ai'
]

def classify_repository_heuristic(
    repo_url: str,
    classifier: Union[str, Dict[str, Dict[str, int]]],
    top_n: int = 3
) -> Dict[str, float]:
    """
    Classify a GitHub repository using keyword-based heuristic method.
    
    This function fetches the README content from a GitHub repository and uses a keyword-based
    heuristic approach to classify it into one or more project types.
    
    Args:
        repo_url: GitHub repository URL.
                 This is a required parameter that specifies the repository to classify.
                 Example: "https://github.com/username/repo-name"
        
        classifier: Classifier name (str) or configuration dictionary (Dict).
                   This is a required parameter that determines what project types to classify against.
                   - If string, looks up the classifier in the registry (e.g., CLASSIFIER_NAMES.PHP)
                   - If dictionary, uses it directly as configuration
                   Example dictionary format:
                   {
                       "Web Framework": {"laravel": 10, "routing": 5, "mvc": 5},
                       "CMS": {"content": 8, "management": 5, "admin": 5}
                   }
        
        top_n: Number of top types to return.
              This is an optional parameter that controls how many top-scoring types to include in the result.
              Default: 3
              Must be a positive integer.
        
    Returns:
        A dictionary mapping project types to confidence scores (0.0 to 1.0).
        Only the top N types are included in the result.
        Example: {"Web Framework": 0.85, "CMS": 0.35, "E-commerce": 0.15}
        
    Raises:
        ValueError: If required parameters are missing or invalid, README cannot be found,
                   or classifier is not found.
                   
    Examples:
        >>> # Classify using PHP classifier
        >>> classify_repository_heuristic(
        ...     repo_url="https://github.com/laravel/laravel",
        ...     classifier=CLASSIFIER_NAMES.PHP
        ... )
        {'Web Framework': 0.85, 'CMS': 0.12, 'E-commerce': 0.03}
        
        >>> # Classify using custom configuration
        >>> custom_config = {
        ...     "Data Science": {"pandas": 10, "numpy": 8, "data": 5},
        ...     "Web Scraping": {"scrape": 10, "crawler": 8, "requests": 5}
        ... }
        >>> classify_repository_heuristic(
        ...     repo_url="https://github.com/pandas-dev/pandas",
        ...     classifier=custom_config,
        ...     top_n=2
        ... )
        {'Data Science': 0.92, 'Web Scraping': 0.08}
    """
    # Validate required parameters
    if not repo_url:
        raise ValueError("Repository URL cannot be empty")
    
    if not classifier:
        raise ValueError("Classifier cannot be empty")
    
    # Validate optional parameters
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    
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
    all_scores = classify_readme_heuristic(readme_text, config)
    
    # Return top N scores
    return get_top_n_scores(all_scores, top_n)

def classify_repository_ai(
    repo_url: str,
    classifier: Union[str, List[str]],
    api_url: str,
    model_name: str,
    api_key: str,
    top_n: int = 3,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    timeout: int = 60
) -> Dict[str, float]:
    """
    Classify a GitHub repository using AI service.
    
    This function fetches the README content from a GitHub repository and uses an AI service
    to classify it into one of the provided project types.
    
    Args:
        repo_url: GitHub repository URL.
                 This is a required parameter that specifies the repository to classify.
                 Example: "https://github.com/username/repo-name"
        
        api_key: API key for the AI service.
                This is a required parameter needed for authentication with the AI service.
                For OpenAI, this is your OpenAI API key.
                For DeepSeek, this is your DeepSeek API key.
        
        model_name: The name of the AI model to use.
                   This is a required parameter that specifies which AI model to use.
                   Supported models include:
                   - OpenAI models: "gpt-3.5-turbo", "gpt-4", etc.
                   - DeepSeek models: "deepseek-chat", "deepseek-coder", etc.
        
        classifier: Classifier name (str) or list of project types (List[str]).
                   This is a required parameter that determines what project types to classify against.
                   - If string, looks up project types in the registry (e.g., CLASSIFIER_NAMES.PHP)
                   - If list, uses it directly as project types (e.g., ["Web Framework", "CMS"])
        
        api_url: Custom API URL for the AI service.
                This is a required parameter that specifies the endpoint for the AI service.
                For GPT models: "https://api.openai.com/v1/chat/completions"
                For DeepSeek models: "https://api.deepseek.com/v1/chat/completions"
        
        top_n: Number of top types to return.
              This is an optional parameter that controls how many top-scoring types to include in the result.
              Default: 3
              Must be a positive integer.
        
        temperature: Controls randomness in the AI response.
                    This is an optional parameter that affects how deterministic the AI's response is.
                    Lower values make the output more deterministic and focused.
                    Higher values make the output more creative and varied.
                    Range: 0.0 to 1.0
                    Default: 0.1
        
        max_tokens: Maximum number of tokens in the AI response.
                   This is an optional parameter that limits the length of the AI's response.
                   If None, the model's default maximum is used.
                   Default: None
                   
        timeout: Request timeout in seconds.
                This is an optional parameter that controls how long to wait for the AI service to respond.
                Default: 60
                Must be a positive integer.
        
    Returns:
        A dictionary mapping project types to confidence scores (0.0 to 1.0).
        Only the top N types are included in the result.
        Example: {"Web Framework": 0.92, "Library": 0.45}
        
    Raises:
        ValueError: If required parameters are missing or invalid, README cannot be found, 
                   classifier is not found, or AI service returns an error.
        
    Examples:
        >>> # Classify using OpenAI with PHP classifier
        >>> classify_repository_ai(
        ...     repo_url="https://github.com/laravel/laravel",
        ...     api_key="your-openai-api-key",
        ...     model_name="gpt-3.5-turbo",
        ...     classifier=CLASSIFIER_NAMES.PHP,
        ...     api_url="https://api.openai.com/v1/chat/completions"
        ... )
        {'Web Framework': 0.95, 'CMS': 0.03, 'E-commerce': 0.02}
        
        >>> # Classify using DeepSeek with custom project types
        >>> classify_repository_ai(
        ...     repo_url="https://github.com/tensorflow/tensorflow",
        ...     api_key="your-deepseek-api-key",
        ...     model_name="deepseek-chat",
        ...     classifier=["Machine Learning Library", "Data Science Tool", "Neural Network"],
        ...     api_url="https://api.deepseek.com/v1/chat/completions",
        ...     temperature=0.2,
        ...     timeout=120
        ... )
        {'Machine Learning Library': 0.89, 'Neural Network': 0.10, 'Data Science Tool': 0.01}
    """
    # Validate required parameters
    if not repo_url:
        raise ValueError("Repository URL cannot be empty")
    
    if not api_key:
        raise ValueError("API key cannot be empty")
    
    if not model_name:
        raise ValueError("Model name cannot be empty")
    
    if not api_url:
        raise ValueError("API URL cannot be empty")
    
    if not classifier:
        raise ValueError("Classifier cannot be empty")
    
    # Validate optional parameters
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    
    if temperature < 0.0 or temperature > 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    
    if timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    
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
    all_scores = classify_readme_ai(
        readme_text, 
        repo_url, 
        api_key, 
        api_url, 
        model_name,
        project_types,
        temperature,
        max_tokens,
        timeout
    )
    
    # Return top N scores
    return get_top_n_scores(all_scores, top_n) 