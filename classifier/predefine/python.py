"""
Python project type configuration.

This module defines project types and keyword weights for Python repositories.
"""

# Python project types and their keyword weight mappings
PYTHON_PROJECT_TYPES = {
    "Web Framework": {
        "django": 10,
        "flask": 10,
        "fastapi": 10,
        "web framework": 8,
        "wsgi": 5,
        "asgi": 5,
        "pyramid": 8,
        "tornado": 8,
        "bottle": 8,
        "web server": 5
    },
    "Data Science": {
        "data science": 10,
        "machine learning": 10,
        "deep learning": 10,
        "numpy": 8,
        "pandas": 8,
        "scikit-learn": 8,
        "tensorflow": 8,
        "pytorch": 8,
        "keras": 8,
        "jupyter": 7,
        "notebook": 5,
        "data analysis": 8,
        "data visualization": 8,
        "matplotlib": 7,
        "seaborn": 7
    },
    "CLI Tool": {
        "command line": 10,
        "cli": 10,
        "terminal": 8,
        "console": 8,
        "argparse": 5,
        "click": 5,
        "typer": 5,
        "shell": 5,
        "command-line": 10,
        "commandline": 10
    },
    "Library/Package": {
        "library": 10,
        "package": 10,
        "module": 8,
        "pip": 5,
        "pypi": 5,
        "dependency": 5,
        "helper": 5,
        "utility": 5,
        "toolkit": 8
    },
    "Web Scraping": {
        "scraping": 10,
        "crawler": 10,
        "spider": 10,
        "beautifulsoup": 8,
        "requests": 5,
        "selenium": 5,
        "web scraper": 10,
        "html parser": 8,
        "data extraction": 8
    },
    "API/Backend": {
        "api": 10,
        "rest": 10,
        "restful": 10,
        "graphql": 10,
        "backend": 10,
        "microservice": 8,
        "server": 5,
        "endpoint": 8,
        "http": 5,
        "json": 5
    },
    "Desktop Application": {
        "desktop": 10,
        "gui": 10,
        "tkinter": 8,
        "pyqt": 8,
        "pyside": 8,
        "wxpython": 8,
        "kivy": 8,
        "qt": 7,
        "user interface": 5,
        "desktop app": 10
    },
    "Testing Tool": {
        "testing": 10,
        "test": 8,
        "pytest": 10,
        "unittest": 10,
        "mock": 8,
        "assertion": 5,
        "test suite": 8,
        "test case": 8,
        "test runner": 8
    }
}

# Export project type names for AI classification
PYTHON_PROJECT_TYPE_NAMES = list(PYTHON_PROJECT_TYPES.keys()) 