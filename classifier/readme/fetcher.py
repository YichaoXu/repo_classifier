"""
README fetching functionality.

This module provides functions to fetch README files from GitHub repositories.
"""

import requests

__all__ = ['get_repo_readme']

def get_repo_readme(repo_url: str) -> str:
    """
    Get README content from a GitHub repository.
    
    Args:
        repo_url: GitHub repository URL
        
    Returns:
        README content as string
        
    Raises:
        ValueError: If repository URL is invalid or README cannot be found
        requests.RequestException: For network-related errors
    """
    # Extract owner and repo from URL
    if not repo_url.startswith("https://github.com/"):
        raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
    
    repo_path = repo_url.replace("https://github.com/", "")
    parts = repo_path.split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid repository URL format: {repo_url}")
    
    owner, repo = parts[0], parts[1]
    
    # Try to download README with different file names and branches
    readme_variants = [
        # master branch variants
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.rst",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README",
        # main branch variants
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.rst",
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/README"
    ]
    
    try:
        for url in readme_variants:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return response.text
            except requests.RequestException:
                # Continue trying other variants
                continue
        
        # If we get here, no README was found
        raise ValueError(f"README not found for repository: {repo_url}")
    
    except requests.RequestException as e:
        raise ValueError(f"Network error when fetching README for {repo_url}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error downloading README for {repo_url}: {str(e)}") 