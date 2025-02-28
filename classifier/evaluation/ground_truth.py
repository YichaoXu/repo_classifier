"""
Ground truth data management and evaluation.

This module provides functionality for managing ground truth data for repository classification
and evaluating classifier performance against this data.
"""

import os
import csv
from typing import Dict, List, Tuple, Optional, Callable, Any

# Default path for ground truth CSV file
DEFAULT_GROUND_TRUTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data',
    'ground_truth.csv'
)

def load_ground_truth(csv_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load ground truth data from a CSV file.
    
    This function loads repository classification ground truth data from a CSV file.
    The CSV should have at least the following columns:
    - repo_url: The GitHub repository URL
    - true_type: The actual/correct project type
    
    Args:
        csv_path: Path to the CSV file containing ground truth data.
                 If None, uses the default path.
                 
    Returns:
        A list of dictionaries containing the ground truth data.
        Each dictionary represents a row in the CSV with column names as keys.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist and cannot be created
        ValueError: If the CSV file is missing required columns
    """
    # Use default path if none provided
    if csv_path is None:
        csv_path = DEFAULT_GROUND_TRUTH_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        # Create empty CSV file with headers if it doesn't exist
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['repo_url', 'true_type', 'notes'])
        return []
    
    # Load CSV file
    data = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        if reader.fieldnames:
            required_columns = ['repo_url', 'true_type']
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"Ground truth CSV is missing required columns: {', '.join(missing_columns)}")
        
        # Read all rows
        for row in reader:
            data.append(row)
    
    return data

def save_ground_truth(data: List[Dict[str, str]], csv_path: Optional[str] = None) -> None:
    """
    Save ground truth data to a CSV file.
    
    Args:
        data: List of dictionaries containing ground truth data
        csv_path: Path to save the CSV file. If None, uses the default path.
    """
    # Use default path if none provided
    if csv_path is None:
        csv_path = DEFAULT_GROUND_TRUTH_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Get all field names from the data
    fieldnames = set()
    for entry in data:
        fieldnames.update(entry.keys())
    
    # Ensure required fields are included
    required_fields = ['repo_url', 'true_type', 'notes']
    fieldnames = required_fields + [f for f in fieldnames if f not in required_fields]
    
    # Save data to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def add_ground_truth_entry(
    repo_url: str,
    true_type: str,
    notes: Optional[str] = None,
    csv_path: Optional[str] = None
) -> None:
    """
    Add a new entry to the ground truth database.
    
    Args:
        repo_url: GitHub repository URL
        true_type: The actual/correct project type
        notes: Optional notes about this entry
        csv_path: Path to the CSV file. If None, uses the default path.
    """
    # Load existing ground truth data
    data = load_ground_truth(csv_path)
    
    # Check if repo_url already exists
    for entry in data:
        if entry['repo_url'] == repo_url:
            # Update existing entry
            entry['true_type'] = true_type
            if notes is not None:
                entry['notes'] = notes
            break
    else:
        # Add new entry if not found
        new_entry = {'repo_url': repo_url, 'true_type': true_type}
        if notes is not None:
            new_entry['notes'] = notes
        data.append(new_entry)
    
    # Save updated ground truth data
    save_ground_truth(data, csv_path)

def get_ground_truth_repos(csv_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get a dictionary of repositories and their true types from ground truth data.
    
    Args:
        csv_path: Path to the CSV file. If None, uses the default path.
        
    Returns:
        Dictionary mapping repository URLs to their true types
    """
    data = load_ground_truth(csv_path)
    return {entry['repo_url']: entry['true_type'] for entry in data}

def evaluate_classifier(
    classifier_func: Callable[[str, Any], Dict[str, float]],
    classifier_args: Any = None,
    csv_path: Optional[str] = None,
    top_n: int = 1
) -> Dict[str, Any]:
    """
    Evaluate a classifier against ground truth data.
    
    This function runs a classifier on repositories from the ground truth database
    and calculates accuracy metrics by comparing the results with the true types.
    
    Args:
        classifier_func: Function that classifies repositories
                        Should take a repo_url as first argument and return a dict of scores
        classifier_args: Additional arguments to pass to the classifier function
        csv_path: Path to the CSV file. If None, uses the default path.
        top_n: Consider classification correct if true type is in top N results
        
    Returns:
        Dictionary with evaluation metrics:
        - accuracy: Overall accuracy (correct / total)
        - correct: Number of correctly classified repositories
        - total: Total number of evaluated repositories
        - detailed: Dictionary with per-repository results
    """
    # Load ground truth data
    data = load_ground_truth(csv_path)
    
    # Initialize results
    results = {
        'accuracy': 0.0,
        'correct': 0,
        'total': len(data),
        'detailed': {}
    }
    
    # Skip evaluation if no ground truth data
    if len(data) == 0:
        return results
    
    # Evaluate each repository
    for entry in data:
        repo_url = entry['repo_url']
        true_type = entry['true_type']
        
        try:
            # Run classifier
            if classifier_args is None:
                scores = classifier_func(repo_url)
            else:
                scores = classifier_func(repo_url, classifier_args)
            
            # Get top N types
            top_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_type_names = [t[0] for t in top_types]
            
            # Check if true type is in top N
            is_correct = true_type in top_type_names
            if is_correct:
                results['correct'] += 1
            
            # Store detailed result
            results['detailed'][repo_url] = {
                'true_type': true_type,
                'predicted': top_type_names,
                'scores': {t: s for t, s in top_types},
                'correct': is_correct
            }
        except Exception as e:
            # Handle errors
            results['detailed'][repo_url] = {
                'true_type': true_type,
                'error': str(e),
                'correct': False
            }
    
    # Calculate accuracy
    if results['total'] > 0:
        results['accuracy'] = results['correct'] / results['total']
    
    return results 