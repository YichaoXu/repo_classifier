"""
AI-based classification functionality.

This module provides AI-powered classification for repositories.
"""

import re
import json
import requests
from typing import Dict, List, Optional, Union
from ..registry import get_classifier, get_available_classifiers

# Export functions
__all__ = ['classify_description_ai']

def classify_description_ai(
    readme_text: str,
    classifier: Union[str, List[str]],
    api_url: str,
    model_name: str,
    api_key: str,
    temperature: float = 0.1,
    max_in_tokens: Optional[int] = 100,
    max_out_tokens: Optional[int] = None,
    timeout: int = 60
) -> Dict[str, float]:
    """
    Classify a repository using AI service based on its README content.
    
    This function sends the README content to an AI service (like OpenAI GPT or DeepSeek)
    and asks it to classify the repository into one of the provided project types.
    
    Args:
        readme_text: The README content of the repository to classify.
                    This should be the raw text content of the README file.
        
        api_key: API key for the AI service.
                This is required for authentication with the AI service.
                For OpenAI, this is your OpenAI API key.
        
        api_url: Custom API URL for the AI service.
                For GPT models: "https://api.openai.com/v1/chat/completions"
                For DeepSeek models: "https://api.deepseek.com/v1/chat/completions"
        
        model_name: The name of the AI model to use.
                   Supported models include:
                   - OpenAI models: "gpt-3.5-turbo", "gpt-4", etc.
                   - DeepSeek models: "deepseek-chat", etc.
        
        classifier: List of possible project types for classification.
                      Example: ["Web Framework", "Library", "CLI Tool"]
        
        temperature: Controls randomness in the AI response.
                    Lower values make the output more deterministic.
                    Range: 0.0 to 1.0
                    Default: 0.1
        
        max_tokens: Maximum number of tokens in the AI response.
                   If None, the model's default maximum is used.
                   
        timeout: Request timeout in seconds.
                Default: 60
    
    Returns:
        A dictionary mapping project types to confidence scores (0.0 to 1.0).
        The primary classification will have the highest score, while other
        types (if provided in project_types) will have minimal scores.
        Example: {"Web Framework": 0.92, "Library": 0.01, "CLI Tool": 0.01}
    
    Raises:
        ValueError: If required parameters are missing or invalid, API request fails,
                   response format is invalid, or model is unsupported.
        requests.RequestException: For network-related errors.
    """
    # Validate required parameters
    if not readme_text:
        raise ValueError("README text cannot be empty")
    
    if not api_key:
        raise ValueError("API key cannot be empty")
    
    if not api_url:
        raise ValueError("API URL cannot be empty")
    
    if not model_name:
        raise ValueError("Model name cannot be empty")
    
    # Parse classifier parameter
    if isinstance(classifier, str):
        # Get classifier from registry by name
        classifier = get_classifier(classifier)
        if not classifier:
            available = get_available_classifiers()
            raise ValueError(f"Classifier not found: {classifier}. Available classifiers: {', '.join(available)}")
    else:
        # Use dictionary directly as configuration
        classifier = classifier
    
    # Validate model name
    supported_models = {
        "gpt": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        "deepseek": ["deepseek-chat", "deepseek-coder"]
    }
    
    model_type = None
    for type_prefix, models in supported_models.items():
        if any(model in model_name.lower() for model in [type_prefix] + [m.lower() for m in models]):
            model_type = type_prefix
            break
    
    if not model_type:
        supported_model_list = [m for models in supported_models.values() for m in models]
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {', '.join(supported_model_list)}")
    
    # Truncate README if too long (to avoid token limits)
    if max_in_tokens and len(readme_text) > max_in_tokens * 4:
        readme_text = readme_text[:max_in_tokens*4] + "...."
    
    # Build prompt
    prompt = f"""
    Analyze the following GitHub repository README and classify it as one of the following project types:
    {', '.join(classifier)}
        
    README:
    {readme_text}
    
    Respond with a JSON object with the following properties:
    - project_type: The classification from the list above
    - confidence: Numerical confidence score between 0-100
    - reasoning: Brief explanation for the classification
    
    JSON response:
    """
    
    try:
        # Different handling based on model type
        if model_type == "deepseek":
            # DeepSeek API implementation
            response = requests.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    **({"max_tokens": max_out_tokens} if max_out_tokens else {})
                },
                timeout=timeout
            )
        elif model_type == "gpt":
            # OpenAI API implementation
            response = requests.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    **({"max_tokens": max_out_tokens} if max_out_tokens else {})
                },
                timeout=timeout
            )
        
        # Process response
        if response.status_code != 200:
            error_message = f"API error: {response.status_code}"
            try:
                error_details = response.json()
                if "error" in error_details:
                    error_message += f" - {error_details['error']['message']}"
            except:
                error_message += f" - {response.text}"
            raise ValueError(error_message)
        
        response_data = response.json()
        
        # Extract content from response (handle different API response formats)
        if "choices" in response_data and len(response_data["choices"]) > 0:
            if "message" in response_data["choices"][0]:
                content = response_data["choices"][0]["message"].get("content", "")
            elif "text" in response_data["choices"][0]:
                content = response_data["choices"][0].get("text", "")
            else:
                raise ValueError(f"Unexpected API response format: {response_data}")
        else:
            raise ValueError(f"No choices found in API response: {response_data}")
        
        # Extract JSON from response
        json_match = re.search(r'({.*})', content, re.DOTALL)
        
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                # Validate response format
                if "project_type" not in result:
                    raise ValueError(f"Missing 'project_type' in AI response: {result}")
                
                if "confidence" not in result:
                    # If confidence is missing but we have a project_type, assume high confidence
                    result["confidence"] = 90
                
                # Convert to score dictionary
                project_type = result["project_type"]
                confidence = float(result["confidence"]) / 100.0  # Normalize to 0-1
                
                # Ensure confidence is in valid range
                confidence = max(0.0, min(1.0, confidence))
                
                # Create scores dictionary with the classified type
                scores = {project_type: confidence}
                
                # Add minimal scores for other types
                for pt in classifier:
                    if pt != project_type:
                        scores[pt] = 0.01
                
                return scores
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON response: {content}")
        else:
            raise ValueError(f"No JSON found in response: {content}")
    
    except requests.RequestException as e:
        raise ValueError(f"Network error when calling AI service: {str(e)}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Unexpected error during AI classification: {str(e)}") 