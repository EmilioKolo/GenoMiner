#!/usr/bin/env python3
"""
Setup script for the SRA to Features Pipeline.
This is a fallback for editable installations when pyproject.toml doesn't work.
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="sra-to-features-pipeline",
        version="1.0.0",
        description="Bioinformatics pipeline for extracting features from SRA data for LLM training",
        author="Bioinformatics Team",
        author_email="team@example.com",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.26.0",
            "pandas>=2.1.0",
            "requests>=2.31.0",
            "cnvpytor>=1.3.0",
            "pydantic>=2.5.0",
            "pydantic-settings>=2.1.0",
            "structlog>=23.2.0",
            "psutil>=5.9.0",
            "click>=8.0.0",
            "rich>=13.0.0",
        ],
        extras_require={
            "dev": [
                "pytest>=7.4.0",
                "pytest-cov>=4.1.0",
                "pytest-mock>=3.12.0",
                "black>=23.11.0",
                "flake8>=6.1.0",
                "mypy>=1.7.0",
                "memory-profiler>=0.61.0",
            ],
            "docs": [
                "sphinx>=7.2.0",
                "sphinx-rtd-theme>=1.3.0",
            ],
        },
        entry_points={
            "console_scripts": [
                "sra-pipeline=sra_pipeline.cli:main",
                "sra-pipeline-setup=sra_pipeline.prepare_files:main",
            ],
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
        ],
    ) 