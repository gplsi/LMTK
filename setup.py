#!/usr/bin/env python
# setup.py - Provides pip installation compatibility alongside Poetry
import os
import re
from setuptools import setup, find_packages

# Function to parse version from pyproject.toml
def get_version_from_pyproject():
    with open("pyproject.toml", "r") as f:
        content = f.read()
    version_match = re.search(r'version = ["\']([^"\']+)["\']', content)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found in pyproject.toml")

# Function to read requirements.txt
def read_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description from README
def read_long_description():
    if os.path.exists("README.md"):
        with open("README.md", "r") as f:
            return f.read()
    return ""

# Read requirements and filter test dependencies
requirements = []
dev_requirements = []
for req in read_requirements():
    if any(req.startswith(test_dep) for test_dep in [
        "pytest", "hypothesis", "sentencepiece"
    ]):
        dev_requirements.append(req)
    else:
        requirements.append(req)

setup(
    name="continual-pretrain",
    version=get_version_from_pyproject(),
    description="Machine learning training framework",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourusername/continual-pretrain",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10,<3.11",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": ["flash-attn"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)