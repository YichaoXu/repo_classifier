"""
AI-based classification functionality.

This module provides AI-powered classification for repositories.
"""

from typing import Dict, List, Optional
import re
import json
import requests

# Export functions
__all__ = ['ai_classify']

def ai_classify(
    readme_text: str,
    repo_url: str,
    api_key: str,
    api_url: Optional[str] = None,
    model_name: str = "gpt-3.5-turbo",
    project_types: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Classify repository using AI service.
    
    Args:
        readme_text: README content
        repo_url: Repository URL for context
        api_key: API key for AI service
        api_url: API URL for AI service
        model_name: AI model name
        project_types: List of possible project types
    
    Returns:
        Dictionary of project types and their confidence scores
    
    Raises:
        ValueError: If API request fails, response format is invalid, or model is unsupported
        requests.RequestException: For network-related errors
    """
    # Truncate README if too long
    if len(readme_text) > 12000:
        readme_text = readme_text[:12000] + "..."
    
    # Build prompt
    prompt = f"""
    Analyze the following GitHub repository README and classify it as one of the following project types:
    {', '.join(project_types) if project_types else 'Determine appropriate project type'}
    
    Repository: {repo_url}
    
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
        if "deepseek" in model_name.lower():
            # DeepSeek API implementation
            api_url = api_url or "https://api.deepseek.com/v1/chat/completions"
            response = requests.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=60
            )
        elif "gpt" in model_name.lower():
            # OpenAI API implementation
            api_url = api_url or "https://api.openai.com/v1/chat/completions"
            response = requests.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=60
            )
        else:
            # Generic implementation for other models
            raise ValueError(f"Unsupported model: {model_name}")
        
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
        
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract JSON from response
        json_match = re.search(r'({.*})', content, re.DOTALL)
        
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                # Convert to score dictionary
                if "project_type" in result and "confidence" in result:
                    project_type = result["project_type"]
                    confidence = result["confidence"] / 100.0  # Normalize to 0-1
                    
                    # Create scores dictionary with the classified type
                    scores = {project_type: confidence}
                    
                    # Add minimal scores for other types if provided
                    if project_types:
                        for pt in project_types:
                            if pt != project_type:
                                scores[pt] = 0.01
                    
                    return scores
                else:
                    raise ValueError("Invalid response format: missing project_type or confidence")
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