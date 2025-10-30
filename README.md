# ML Analysis Package

A modular Python package for machine learning analysis with Random Forest classification, feature selection, permutation testing, and comprehensive visualization.

## 📁 Package Structure

```
ml-analysis-package/
├── pyproject.toml              # Modern package configuration
├── requirements.txt            # Dependency list
├── README.md                   # This file
│
├── src/
│   ├── preprocessing/
│   │   ├── data_processing.py      # Data loading with imputation
│   │   └── feature_selection.py    # SelectKBest and feature utilities
│   │
│   ├── models/
│   │   ├── random_forest.py        # RF classifier with CV
│   │   └── evaluation.py           # Metrics and evaluation
│   │
│   ├── analysis/
│   │   ├── importance.py           # Feature importance methods
│   │   └── statistical_tests.py    # Permutation tests
│   │
│   ├── visualization/
│   │   ├── model_plots.py          # Confusion matrix, importance
│   │   ├── data_plots.py           # Trends, boxplots, heatmaps
│   │   └── dimension_reduction.py  # t-SNE, UMAP, SHAP
│   │
│   └── utils/
│       ├── data_utils.py           # Data preparation helpers
│       └── io_utils.py             # Save/load utilities
│
└── examples/
    ├── complete_workflow.py        # Full analysis (sys.path version)
    └── modern_workflow.py          # Using installed package
```

## 🚀 Installation

### Quick Install (Recommended)

```bash
# Navigate to package directory
cd /path/to/ml-analysis-package

# Install in editable mode with all dependencies
pip install -e ".[all]"

# Verify installation
python -c "from preprocessing import load_and_impute; print('✅ Ready!')"
```

### Installation Options

```bash
# Basic installation (required dependencies only)
pip install -e .

# With UMAP support
pip install -e ".[umap]"

# With ordinal regression support
pip install -e ".[ordinal]"

# With development tools (black, flake8, mypy)
pip install -e ".[dev]"

# Everything (UMAP + ordinal + dev tools)
pip install -e ".[all]"
```

### Using Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
# OR
venv\Scripts\activate             # Windows

# Install package
pip install -e ".[all]"
```

## 📖 Quick Start

### Basic Usage

```python
from preprocessing import load_and_impute, select_k_best_features
from models import train_evaluate_model
from utils import prepare_data

# Load and prepare data
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, y_raw, y, le, _ = prepare_data(df, target_col='Groups')

# Select features and train
X_selected, features, _ = select_k_best_features(X, y, k=15)
clf = train_evaluate_model(X_selected, y)

print(f"CV Accuracy: {clf.cv_results['mean_accuracy']:.3f}")
```

### Complete Analysis Pipeline

See `examples/modern_workflow.py` for a comprehensive workflow including:
- ✅ Data loading with imputation
- ✅ Feature selection
- ✅ Model training with cross-validation  
- ✅ Permutation testing for significance
- ✅ Feature importance analysis
- ✅ SHAP interpretation
- ✅ Visualization (confusion matrix, trends, t-SNE, UMAP)
- ✅ Automated result saving

Run it:
```bash
cd examples
python modern_workflow.py
```

## 💡 Common Workflows

### Quick Feature Screening

```python
from preprocessing import load_and_impute, select_k_best_features, get_top_n_features
from models import train_evaluate_model
from analysis import get_feature_importance_df
from visualization import plot_compound_trends
from utils import prepare_data

# Load and prepare
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, _, y, le, _ = prepare_data(df)

# Select and train
X_sel, features, _ = select_k_best_features(X, y, k=15)
clf = train_evaluate_model(X_sel, y)

# Get important features
importance_df = get_feature_importance_df(clf.model, features)
top_5 = get_top_n_features(importance_df, n=5)

# Visualize
plot_compound_trends(df, top_5, scale='log')
```

### Ordinal Regression (For Ordered Outcomes)

```python
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from visualization.ordinal_plots import plot_ordinal_coefficients, plot_ordinal_confusion_matrix
from utils import prepare_data

# Load data
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, y_raw, _, _, _ = prepare_data(df)

# Encode as ordinal: Green=0, Ripe=1, Overripe=2
y = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])

# Train ordinal model
results = train_evaluate_ordinal(X, y, model_type='LogisticAT')

# Visualize coefficients
plot_ordinal_coefficients(results['coefficients'], top_n=10)
plot_ordinal_confusion_matrix(results['all_true'], results['all_pred'])

print(f"Accuracy: {results['mean_accuracy']:.3f}")
print(f"MAE: {results['mean_mae']:.3f}")
```

### Rigorous Validation with Permutation Test

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from analysis import permutation_test
from visualization import plot_confusion_matrix, plot_permutation_test_distribution

# Train model
clf = train_evaluate_model(X, y)

# Permutation test for significance
cv = StratifiedKFold(n_splits=4, shuffle=False)
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
perm_results = permutation_test(rf, X, y, cv, n_permutations=1000)

# Visualize
plot_confusion_matrix(clf.confusion_matrix, le.classes_)
plot_permutation_test_distribution(
    perm_results['perm_accuracies'],
    perm_results['observed_accuracy']
)

print(f"p-value: {perm_results['p_value']:.4f}")
```

### Flexible Top-N Feature Analysis

```python
from preprocessing import get_top_n_features
from analysis import permutation_test_with_feature_subset
from visualization import plot_compound_trends

# Analyze different numbers of features
for n in [3, 5, 10]:
    top_n = get_top_n_features(importance_df, n=n)
    X_subset = X_df[top_n]
    
    # Train and test
    clf = train_evaluate_model(X_subset, y)
    perm_results = permutation_test_with_feature_subset(
        rf, X_df, top_n, y, n_permutations=1000
    )
    
    # Visualize
    plot_compound_trends(df, top_n, scale='linear')
    print(f"Top {n}: Accuracy={clf.cv_results['mean_accuracy']:.3f}, p={perm_results['p_value']:.4f}")
```

## 📚 Module Reference

### preprocessing
- `load_and_impute()` - Load CSV with median imputation by group
- `select_k_best_features()` - Select top K features using statistical tests
- `get_top_n_features()` - Extract top N feature names from importance DataFrame

### models
- `train_evaluate_model()` - Train RF with cross-validation (convenience function)
- `RFClassifierCV` - Random Forest classifier with built-in CV support
- `compare_models()` - Compare multiple model results
- `OrdinalClassifier` - Ordinal regression for ordered outcomes (requires `mord`)
- `train_evaluate_ordinal()` - Train ordinal model with CV
- `compare_ordinal_models()` - Compare different ordinal model types

### analysis
- `get_feature_importance_df()` - Get tree-based feature importance
- `compute_permutation_importance()` - Compute permutation-based importance
- `permutation_test()` - Test model significance vs random labels
- `get_shap_values()` - Compute SHAP values for interpretability

### visualization
- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_feature_importance()` - Feature importance bar chart
- `plot_compound_trends()` - Line plots across groups (linear/log scale)
- `plot_compound_boxplots()` - Distribution boxplots by group
- `plot_tsne()` - t-SNE dimensionality reduction
- `plot_umap()` - UMAP dimensionality reduction
- `plot_shap_summary()` - SHAP summary plots per class

### utils
- `prepare_data()` - Separate features/target and encode labels
- `print_data_summary()` - Print dataset statistics
- `create_results_directory()` - Create organized output folders
- `save_results()` - Save results to JSON/pickle

## 🔧 Configuration Tips

### Model Hyperparameters
- `max_depth=3` - Good for small datasets (prevents overfitting)
- `n_estimators=100` - Balances performance and speed
- `n_splits=None` - Auto-detects based on minimum class size

### Feature Selection
- Start with `k=15-20` features
- Narrow down to top `3-5` for interpretation
- Use log scale for compound visualization with wide value ranges

### Permutation Testing
- Use `n_permutations=1000` for reliable p-values
- p < 0.05 indicates significant relationship between features and labels

## 📊 Output Structure

Using `create_results_directory()` creates organized folders:

```
results/
└── your_analysis_name/
    ├── figures/
    │   ├── confusion_matrix.png
    │   ├── feature_importance.png
    │   ├── compound_trends_log.png
    │   └── ...
    ├── data/
    │   ├── selected_features.csv
    │   ├── feature_importance.csv
    │   └── model_comparison.csv
    ├── models/
    │   └── rf_model.pkl
    └── results.json
```

## 🛠️ Alternative Usage (Without Installation)

If you prefer not to install the package:

```python
import sys
sys.path.insert(0, 'src')

from preprocessing.data_processing import load_and_impute
from models.random_forest import train_evaluate_model
# ... etc
```

**Note**: Installing with `pip install -e .` is strongly recommended for cleaner imports.

## 🔍 Troubleshooting

### Import Errors
```bash
# Check if installed
pip list | grep ml-analysis

# Reinstall
pip install -e . --force-reinstall
```

### UMAP Not Found
```bash
pip install umap-learn
```

### SHAP Installation Issues
```bash
pip install shap --no-cache-dir
```

## 📝 Dependencies

**Required:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- shap >= 0.40.0

**Optional:**
- umap-learn >= 0.5.0 (for UMAP visualization)
- mord >= 0.6 (for ordinal regression)

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions welcome! Please ensure code is formatted with `black` before submitting.

## 📧 Contact

[Your Contact Information]

---

**Version**: 0.1.0  
**Python**: >= 3.8