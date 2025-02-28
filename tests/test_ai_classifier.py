"""
Tests for the AI classifier.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import classifier
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier import (
    classify_readme_ai,
    classify_repository_ai
)

class TestAIClassifier(unittest.TestCase):
    """Test cases for the AI classifier."""
    
    @patch('classifier.ai_classifier.requests.post')
    def test_ai_classify_gpt(self, mock_post):
        """Test AI classification with GPT model."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"project_type": "Web Framework", "confidence": 90, "reasoning": "Test reasoning"}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Test data
        readme_text = "This is a test README"
        repo_url = "https://github.com/test/repo"
        api_key = "test_api_key"
        project_types = ["Web Framework", "Library", "CLI Tool"]
        
        # Call function
        result = classify_readme_ai(
            readme_text,
            repo_url,
            api_key,
            model_name="gpt-3.5-turbo",
            project_types=project_types
        )
        
        # Assertions
        self.assertIn("Web Framework", result)
        self.assertEqual(result["Web Framework"], 0.9)  # 90/100
        self.assertIn("Library", result)
        self.assertIn("CLI Tool", result)
        
        # Check API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test_api_key")
        self.assertEqual(kwargs["json"]["model"], "gpt-3.5-turbo")
    
    @patch('classifier.ai_classifier.requests.post')
    def test_ai_classify_deepseek(self, mock_post):
        """Test AI classification with DeepSeek model."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"project_type": "Data Science", "confidence": 85, "reasoning": "Test reasoning"}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Test data
        readme_text = "This is a test README"
        repo_url = "https://github.com/test/repo"
        api_key = "test_api_key"
        project_types = ["Web Framework", "Data Science", "CLI Tool"]
        
        # Call function
        result = classify_readme_ai(
            readme_text,
            repo_url,
            api_key,
            model_name="deepseek-chat",
            project_types=project_types
        )
        
        # Assertions
        self.assertIn("Data Science", result)
        self.assertEqual(result["Data Science"], 0.85)  # 85/100
        
        # Check API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "deepseek-chat")
    
    @patch('classifier.ai_classifier.requests.post')
    def test_ai_classify_error_handling(self, mock_post):
        """Test AI classification error handling."""
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Error message"
        mock_post.return_value = mock_response
        
        # Test data
        readme_text = "This is a test README"
        repo_url = "https://github.com/test/repo"
        api_key = "test_api_key"
        
        # Call function and check exception
        with self.assertRaises(ValueError) as context:
            classify_readme_ai(
                readme_text,
                repo_url,
                api_key,
                model_name="gpt-3.5-turbo"
            )
        
        self.assertIn("API error: 400", str(context.exception))
    
    @patch('classifier.core.ai_classify')
    @patch('classifier.core.get_repo_readme')
    def test_classify_repository_ai(self, mock_get_readme, mock_ai_classify):
        """Test classify_repository_ai function."""
        # Mock dependencies
        mock_get_readme.return_value = "Test README"
        mock_ai_classify.return_value = {
            "Web Framework": 0.9,
            "Library": 0.5,
            "CLI Tool": 0.3
        }
        
        # Call function
        with patch('classifier.core.ALL_PROJECT_TYPE_NAMES', {'python': ['Web Framework', 'Library', 'CLI Tool']}):
            result = classify_repository_ai(
                "https://github.com/test/repo",
                api_key="test_api_key",
                classifier="python",
                top_n=2
            )
        
        # Assertions
        self.assertEqual(len(result), 2)
        self.assertIn("Web Framework", result)
        self.assertIn("Library", result)
        self.assertNotIn("CLI Tool", result)  # Should be excluded by top_n=2

if __name__ == "__main__":
    unittest.main() 