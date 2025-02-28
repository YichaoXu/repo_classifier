"""
Tests for the heuristic classifier.
"""

import unittest
import sys
import os

# Add parent directory to path to import classifier
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier import (
    classify_repository_heuristic,
    register_classifier,
    get_classifier,
    get_available_classifiers,
    classify_readme_heuristic,
    normalize_scores,
    get_top_n_scores
)

class TestHeuristicClassifier(unittest.TestCase):
    """Test cases for the heuristic classifier."""
    
    def test_normalize_scores(self):
        """Test score normalization."""
        scores = {"A": 10, "B": 5, "C": 0}
        normalized = normalize_scores(scores)
        
        self.assertEqual(normalized["A"], 1.0)
        self.assertEqual(normalized["B"], 0.5)
        self.assertEqual(normalized["C"], 0.0)
        
        # Test empty scores
        self.assertEqual(normalize_scores({}), {})
        
        # Test all zeros
        all_zeros = {"A": 0, "B": 0}
        self.assertEqual(normalize_scores(all_zeros), {"A": 0.0, "B": 0.0})
    
    def test_get_top_n_scores(self):
        """Test getting top N scores."""
        scores = {"A": 0.9, "B": 0.7, "C": 0.8, "D": 0.3}
        
        # Top 2
        top2 = get_top_n_scores(scores, 2)
        self.assertEqual(len(top2), 2)
        self.assertIn("A", top2)
        self.assertIn("C", top2)
        
        # Top 3
        top3 = get_top_n_scores(scores, 3)
        self.assertEqual(len(top3), 3)
        self.assertIn("A", top3)
        self.assertIn("B", top3)
        self.assertIn("C", top3)
        
        # Top more than available
        top10 = get_top_n_scores(scores, 10)
        self.assertEqual(len(top10), 4)  # Only 4 items available
    
    def test_heuristic_classify(self):
        """Test the heuristic classification function."""
        # Simple test with mock README
        readme = """
        This is a web application framework for building modern web apps.
        It includes MVC architecture and follows RESTful principles.
        """
        
        project_types = {
            "Web App": {
                "web application": 10,
                "web app": 10
            },
            "Framework": {
                "framework": 10,
                "mvc": 8
            },
            "API": {
                "api": 10,
                "rest": 8
            }
        }
        
        results = classify_readme_heuristic(readme, project_types)
        
        # Check that all types are present
        self.assertIn("Web App", results)
        self.assertIn("Framework", results)
        self.assertIn("API", results)
        
        # Check that scores are normalized
        self.assertLessEqual(results["Web App"], 1.0)
        self.assertLessEqual(results["Framework"], 1.0)
        self.assertLessEqual(results["API"], 1.0)
    
    def test_register_and_get_classifier(self):
        """Test registering and retrieving classifiers."""
        # Define a test classifier
        test_config = {
            "Type1": {"keyword1": 10, "keyword2": 5},
            "Type2": {"keyword3": 8, "keyword4": 4}
        }
        
        # Register the classifier
        register_classifier("test", test_config)
        
        # Get the classifier
        retrieved = get_classifier("test")
        
        # Check that it matches
        self.assertEqual(retrieved, test_config)
        
        # Check that it's in the available classifiers
        available = get_available_classifiers()
        self.assertIn("test", available)

if __name__ == "__main__":
    unittest.main() 