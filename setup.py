"""
Setup script for jaxEmbeddingMilvus.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="embedding_trident",
    version="0.1.0",
    description="Production-ready image embedding service with JAX, Triton, and Milvus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Noah Zhang",
    author_email="noahzhy@users.noreply.github.com",
    url="https://github.com/noahzhy/EmbeddingTrident",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "embedding-trident=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="jax triton milvus embedding vector-search image-processing",
    project_urls={
        "Bug Reports": "https://github.com/noahzhy/EmbeddingTrident/issues",
        "Source": "https://github.com/noahzhy/EmbeddingTrident",
    },
)
