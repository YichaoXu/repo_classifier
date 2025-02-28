"""
Classifier configuration registry module.

This module provides functionality to register, manage, and retrieve
classifier configurations for repository classification.
"""

from typing import Dict, List, Optional
import importlib.util
import inspect

from .predefine import ALL_PROJECT_TYPES

# Export functions
__all__ = [
    'register_classifier',
    'unregister_classifier',
    'get_classifier',
    'get_available_classifiers',
    'load_classifier_from_module',
    'create_classifier_from_file'
]

# Global classifier configuration registry
_CLASSIFIER_REGISTRY = {
    # Import built-in configurations
    **ALL_PROJECT_TYPES
}

def register_classifier(name: str, config: Dict[str, Dict[str, int]]) -> None:
    """
    Register a new classifier configuration.
    
    Args:
        name: Classifier name
        config: Project types and their keyword weight mappings
    """
    _CLASSIFIER_REGISTRY[name.lower()] = config

def unregister_classifier(name: str) -> bool:
    """
    Remove a registered classifier configuration.
    
    Args:
        name: Classifier name
        
    Returns:
        True if successfully removed, False otherwise
    """
    if name.lower() in _CLASSIFIER_REGISTRY:
        del _CLASSIFIER_REGISTRY[name.lower()]
        return True
    return False

def get_classifier(name: str) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Get a registered classifier configuration.
    
    Args:
        name: Classifier name
        
    Returns:
        Classifier configuration or None if not found
    """
    return _CLASSIFIER_REGISTRY.get(name.lower())

def get_available_classifiers() -> List[str]:
    """
    Get all available classifier names.
    
    Returns:
        List of classifier names
    """
    return list(_CLASSIFIER_REGISTRY.keys())

def load_classifier_from_module(module_path: str, attribute_name: Optional[str] = None) -> str:
    """
    Load classifier configuration from a Python module.
    
    Args:
        module_path: Module path (can be file path or import path)
        attribute_name: Name of attribute to load, if None loads all uppercase dictionaries
        
    Returns:
        Names of registered classifiers
    """
    # Try to load as file path
    if module_path.endswith('.py'):
        module_name = module_path.split('/')[-1].replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    else:
        # Try to load as import path
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            raise ValueError(f"Could not import module: {module_path}")
    
    # Extract and register configurations
    if attribute_name:
        # Load specific attribute
        if hasattr(module, attribute_name):
            config = getattr(module, attribute_name)
            if isinstance(config, dict):
                register_classifier(attribute_name, config)
                return attribute_name
            else:
                raise ValueError(f"Attribute {attribute_name} is not a dictionary")
        else:
            raise ValueError(f"Attribute not found in module: {attribute_name}")
    else:
        # Automatically load all uppercase dictionaries as configurations
        registered = []
        for name, value in inspect.getmembers(module):
            if name.isupper() and isinstance(value, dict):
                if "_PROJECT_TYPES" in name:
                    # Use shorter name (remove _PROJECT_TYPES)
                    short_name = name.replace("_PROJECT_TYPES", "").lower()
                    register_classifier(short_name, value)
                    registered.append(short_name)
                else:
                    register_classifier(name.lower(), value)
                    registered.append(name.lower())
        
        if registered:
            return ", ".join(registered)
        else:
            raise ValueError("No valid configuration dictionaries found in module")

def create_classifier_from_file(file_path: str, encoding: str = 'utf-8') -> Dict[str, Dict[str, int]]:
    """
    Create classifier configuration from a text file.
    
    File format:
    TYPE: Type Name
    keyword1: 10
    keyword2: 8
    
    TYPE: Another Type
    keyword3: 10
    keyword4: 5
    
    Args:
        file_path: Text file path
        encoding: File encoding
        
    Returns:
        Classifier configuration dictionary
    """
    config = {}
    current_type = None
    
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('TYPE:'):
                current_type = line[5:].strip()
                config[current_type] = {}
            elif current_type and ':' in line:
                keyword, weight = line.split(':', 1)
                try:
                    config[current_type][keyword.strip()] = int(weight.strip())
                except ValueError:
                    pass
    
    return config
