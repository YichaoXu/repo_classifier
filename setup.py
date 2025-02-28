from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="repo_classifier",
    version="0.1.1",
    author="YichaoXU",
    author_email="yxu166@jhu.edu",
    description="A library for classifying GitHub repositories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YichaoXu/repo_classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.0.0",
            "mypy>=0.800",
        ],
    },
)
