"""
CrewGraph AI - Production-ready setup configuration
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="crewgraph-ai",
    version="1.0.0",
    author="CrewGraph AI Team",
    author_email="team@crewgraph-ai.com",
    description="Production-ready library combining CrewAI and LangGraph for advanced agent orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/crewgraph/crewgraph-ai",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "crewai>=0.28.0",
        "langgraph>=0.0.40",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "asyncio-mqtt>=0.11.0",
        "structlog>=23.1.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "redis": ["redis>=4.0.0", "hiredis>=2.0.0"],
        "faiss": ["faiss-cpu>=1.7.0", "sentence-transformers>=2.0.0"],
        "visualization": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "networkx>=2.8.0",
            "graphviz>=0.20.0",
            "pandas>=1.5.0",
        ],
        "full": [
            "redis>=4.0.0", 
            "hiredis>=2.0.0",
            "faiss-cpu>=1.7.0", 
            "sentence-transformers>=2.0.0",
            "sqlalchemy>=2.0.0",
            "aiosqlite>=0.19.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "networkx>=2.8.0",
            "graphviz>=0.20.0",
            "pandas>=1.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
            "types-redis>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crewgraph=crewgraph_ai.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)