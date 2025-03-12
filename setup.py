"""
Setup script for SumWiseAI package.
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="sumwiseai",
        version="0.1.0",
        description="Intelligent document processing with advanced chunking and summarization",
        author="Kris Naleszkiewicz",
        author_email="kris.nale@gmail.com",
        packages=find_packages(),  # Find packages in the current directory
        python_requires=">=3.8",
        install_requires=[
            "litellm>=1.2.0",
            "nltk>=3.7.0",
            "numpy>=1.20.0",
        ],
        extras_require={
            "dev": [
                "pytest>=7.0.0",
                "black>=23.1.0",
                "isort>=5.12.0",
                "mypy>=1.0.0",
                "ruff>=0.0.54",
            ],
            "docs": [
                "mkdocs>=1.4.0",
                "mkdocs-material>=8.5.10",
                "mkdocstrings[python]>=0.19.0",
            ],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Text Processing :: Linguistic",
        ],
    )