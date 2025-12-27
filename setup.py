# setup.py
"""
Setup script for Astrophysics Data Augmentation Pipeline
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="astrophysics-augmentation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="VAE + PINN framework for physics-informed data augmentation in astrophysics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/astrophysics-augmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "ipython>=8.12.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "astro-augment=app:main",
        ],
    },
)
