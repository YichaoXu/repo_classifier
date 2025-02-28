"""
Tests for the classifier registry.
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path to import classifier
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier import (
    register_classifier,
    unregister_classifier,
    get_classifier,
    get_available_classifiers,
    load_classifier_from_module,
    create_classifier_from_file
)

class TestRegistry(unittest.TestCase):
    """Test cases for the classifier registry."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Save initial classifiers to restore later
        self.initial_classifiers = get_available_classifiers()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove any test classifiers
        for name in get_available_classifiers():
            if name not in self.initial_classifiers and name.startswith("test_"):
                unregister_classifier(name)
    
    def test_register_and_get_classifier(self):
        """Test registering and retrieving classifiers."""
        # Define a test classifier
        test_config = {
            "Type1": {"keyword1": 10, "keyword2": 5},
            "Type2": {"keyword3": 8, "keyword4": 4}
        }
        
        # Register the classifier
        register_classifier("test_classifier", test_config)
        
        # Get the classifier
        retrieved = get_classifier("test_classifier")
        
        # Check that it matches
        self.assertEqual(retrieved, test_config)
        
        # Check that it's in the available classifiers
        available = get_available_classifiers()
        self.assertIn("test_classifier", available)
    
    def test_unregister_classifier(self):
        """Test unregistering classifiers."""
        # Define and register a test classifier
        test_config = {
            "Type1": {"keyword1": 10, "keyword2": 5}
        }
        register_classifier("test_to_remove", test_config)
        
        # Verify it's registered
        self.assertIn("test_to_remove", get_available_classifiers())
        
        # Unregister it
        result = unregister_classifier("test_to_remove")
        
        # Check result and that it's gone
        self.assertTrue(result)
        self.assertNotIn("test_to_remove", get_available_classifiers())
        
        # Try to unregister non-existent classifier
        result = unregister_classifier("non_existent")
        self.assertFalse(result)
    
    def test_create_classifier_from_file(self):
        """Test creating classifier from text file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write("""
TYPE: Test Type 1
keyword1: 10
keyword2: 8
keyword3: 5

TYPE: Test Type 2
keyword4: 10
keyword5: 7
            """)
            temp_path = temp.name
        
        try:
            # Create classifier from file
            config = create_classifier_from_file(temp_path)
            
            # Check structure
            self.assertIn("Test Type 1", config)
            self.assertIn("Test Type 2", config)
            self.assertEqual(config["Test Type 1"]["keyword1"], 10)
            self.assertEqual(config["Test Type 1"]["keyword2"], 8)
            self.assertEqual(config["Test Type 1"]["keyword3"], 5)
            self.assertEqual(config["Test Type 2"]["keyword4"], 10)
            self.assertEqual(config["Test Type 2"]["keyword5"], 7)
            
            # Register it
            register_classifier("test_from_file", config)
            self.assertIn("test_from_file", get_available_classifiers())
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_classifier_from_module(self):
        """Test loading classifier from Python module."""
        # Create a temporary module file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp:
            temp.write("""
TEST_PROJECT_TYPES = {
    "ModuleType1": {
        "keyword1": 10,
        "keyword2": 5
    },
    "ModuleType2": {
        "keyword3": 8,
        "keyword4": 4
    }
}

ANOTHER_CONFIG = {
    "AnotherType": {
        "keyword5": 10
    }
}
            """)
            temp_path = temp.name
        
        try:
            # Load classifier from module
            result = load_classifier_from_module(temp_path)
            
            # Check that both configs were loaded
            self.assertIn("test", get_available_classifiers())
            self.assertIn("another_config", get_available_classifiers())
            
            # Check specific attribute loading
            unregister_classifier("test")
            unregister_classifier("another_config")
            
            result = load_classifier_from_module(temp_path, "TEST_PROJECT_TYPES")
            self.assertIn("TEST_PROJECT_TYPES", get_available_classifiers())
            self.assertNotIn("ANOTHER_CONFIG", get_available_classifiers())
            
            # Clean up
            unregister_classifier("TEST_PROJECT_TYPES")
        finally:
            # Clean up
            os.unlink(temp_path)

if __name__ == "__main__":
    unittest.main() 