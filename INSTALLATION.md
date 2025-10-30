# Installation Guide

## Quick Install (Editable Mode)

The easiest way to get started is to install the package in editable mode. This allows you to modify the code and see changes immediately without reinstalling.

### 1. Clone or Download the Repository

```bash
cd /path/to/your/package
```

### 2. Install in Editable Mode

```bash
# Basic installation
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### 3. Verify Installation

```python
import ml_analysis
print(ml_analysis.__version__)
```

## Installation Options

### Basic Installation (Required Dependencies Only)

```bash
pip install -e .
```

This installs:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap

### With UMAP Support

```bash
pip install -e ".[umap]"
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

Includes testing and linting tools:
- pytest
- black (code formatter)
- flake8 (linter)
- mypy (type checker)

### Everything

```bash
pip install -e ".[all]"
```

## Usage After Installation

Once installed, you can import modules directly:

```python
# Instead of sys.path hacks, use proper imports
from preprocessing import load_and_impute, select_k_best_features
from models import train_evaluate_model
from visualization import plot_confusion_matrix
from utils import prepare_data, save_results
```

## Creating a Virtual Environment (Recommended)

### Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install package
pip install -e ".[all]"
```

### Using conda

```bash
# Create conda environment
conda create -n ml_analysis python=3.10

# Activate
conda activate ml_analysis

# Install package
pip install -e ".[all]"
```

## Verifying Installation

Run this to check all components:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Check package imports
from preprocessing import load_and_impute, select_k_best_features
from models import RFClassifierCV, train_evaluate_model
from analysis import permutation_test, get_feature_importance_df
from visualization import plot_confusion_matrix, plot_tsne
from utils import prepare_data, save_results

print("✅ All imports successful!")
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. You've activated your virtual environment
2. You're in the correct directory
3. The package is installed: `pip list | grep ml-analysis`

### UMAP Not Found

```bash
pip install umap-learn
```

### SHAP Installation Issues

On some systems, SHAP requires additional dependencies:

```bash
# macOS
brew install cmake

# Linux
sudo apt-get install cmake

# Then install SHAP
pip install shap
```

## Development Setup

For development work:

```bash
# Clone repository
git clone https://github.com/yourusername/ml-analysis-package.git
cd ml-analysis-package

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Check linting
flake8 src/
```

## Updating the Package

If you make changes to the code:

```bash
# In editable mode, changes are reflected immediately
# No need to reinstall

# But if you modify pyproject.toml or add new dependencies:
pip install -e ".[all]" --force-reinstall
```

## Uninstalling

```bash
pip uninstall ml-analysis-package
```

## Building Distribution Packages

To create distributable packages:

```bash
# Install build tools
pip install build

# Build wheel and source distribution
python -m build

# Output will be in dist/
# - ml_analysis_package-0.1.0-py3-none-any.whl
# - ml_analysis_package-0.1.0.tar.gz
```

## Installing from Distribution

```bash
pip install dist/ml_analysis_package-0.1.0-py3-none-any.whl
```

## Requirements File Alternative

If you prefer not to use pyproject.toml:

```bash
pip install -r requirements.txt
```

Then add the src directory to your Python path in each script:

```python
import sys
sys.path.insert(0, 'src')
```

However, using `pip install -e .` is strongly recommended for cleaner imports.