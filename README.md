# ML omics Analysis Package

A modular Python package for machine learning analysis with Random Forest classification, ordinal regression, feature selection, permutation testing, and comprehensive visualization.

## 📁 Package Structure

```
ml-analysis-package/
├── pyproject.toml              # Modern package configuration
├── requirements.txt            # Dependency list
├── README.md                   # This file
│
├── src/
│   ├── preprocessing/          # Data loading & feature selection
│   ├── models/                 # RF, ordinal regression, evaluation
│   ├── analysis/               # Feature importance, permutation tests
│   ├── visualization/          # All plotting functions
│   └── utils/                  # Data preparation, I/O
│
└── examples/
    ├── complete_workflow.py    # Full RF analysis
    ├── modern_workflow.py      # Using installed package
    └── ordinal_workflow.py     # Ordinal regression
```

## 🚀 Installation

```bash
cd /path/to/ML_omics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux: venv\Scripts\activate on Windows

# Install
pip install -e ".[all]"         # Everything
pip install -e .                # Core only
pip install -e ".[umap]"        # + UMAP
pip install -e ".[ordinal]"     # + Ordinal regression
pip install -e ".[dev]"         # + Dev tools
```

Verify: `python -c "from preprocessing import load_and_impute; print('✅')"`

## 📖 Quick Start

### Basic Random Forest Workflow

```python
from preprocessing import load_and_impute, select_k_best_features
from models import train_evaluate_model
from utils import prepare_data

# Load and prepare
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, _, y, le, _ = prepare_data(df, target_col='Groups')

# Select features and train with CV
X_selected, features, _ = select_k_best_features(X, y, k=15)
clf = train_evaluate_model(X_selected, y)

print(f"CV Accuracy: {clf.cv_results['mean_accuracy']:.3f}")
```

### Ordinal Regression (For Ordered Outcomes)

```python
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from visualization.ordinal_plots import plot_ordinal_coefficients

# Load data
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, y_raw, _, _, _ = prepare_data(df)

# Encode as ordinal: Green=0, Ripe=1, Overripe=2
y = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])

# Train with CV
results = train_evaluate_ordinal(X, y)

# Visualize
plot_ordinal_coefficients(results['coefficients'], top_n=10)

print(f"Accuracy: {results['mean_accuracy']:.3f}, MAE: {results['mean_mae']:.3f}")
```

## 💡 Common Workflows

### Feature Screening

```python
from preprocessing import load_and_impute, select_k_best_features, get_top_n_features
from models import train_evaluate_model
from analysis import get_feature_importance_df
from visualization import plot_compound_trends
from utils import prepare_data

df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, _, y, le, _ = prepare_data(df)
X_sel, features, _ = select_k_best_features(X, y, k=15)
clf = train_evaluate_model(X_sel, y)

importance_df = get_feature_importance_df(clf.model, features)
top_5 = get_top_n_features(importance_df, n=5)
plot_compound_trends(df, top_5, scale='log')
```

### Rigorous Validation

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from analysis import permutation_test
from visualization import plot_confusion_matrix, plot_permutation_test_distribution

clf = train_evaluate_model(X, y)

cv = StratifiedKFold(n_splits=4)
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
perm = permutation_test(rf, X, y, cv, n_permutations=1000)

plot_confusion_matrix(clf.confusion_matrix, le.classes_)
plot_permutation_test_distribution(perm['perm_accuracies'], perm['observed_accuracy'])
print(f"p={perm['p_value']:.4f}")
```

### Ordinal Regression

```python
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from visualization.ordinal_plots import plot_ordinal_coefficients

df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, y_raw, _, _, _ = prepare_data(df)
y = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])

results = train_evaluate_ordinal(X, y)
plot_ordinal_coefficients(results['coefficients'], top_n=10)
print(f"Acc={results['mean_accuracy']:.3f}, MAE={results['mean_mae']:.3f}")
```

## 📚 Key Modules

### preprocessing
- `load_and_impute()` - Load CSV with median imputation by group
- `select_k_best_features()` - Select top K features
- `get_top_n_features()` - Extract top N feature names

### models
- `train_evaluate_model()` - Train RF with CV
- `RFClassifierCV` - RF classifier with built-in CV
- `OrdinalClassifier` - Ordinal regression (requires `mord`)
- `train_evaluate_ordinal()` - Train ordinal model with CV

### analysis
- `get_feature_importance_df()` - Tree-based importance
- `compute_permutation_importance()` - Permutation-based importance
- `permutation_test()` - Test model significance
- `get_shap_values()` - SHAP interpretability

### visualization
- `plot_confusion_matrix()` - Confusion matrix
- `plot_feature_importance()` - Importance bar chart
- `plot_compound_trends()` - Line plots (linear/log scale)
- `plot_compound_boxplots()` - Distribution boxplots
- `plot_tsne()`, `plot_umap()` - Dimensionality reduction
- `plot_ordinal_coefficients()` - Ordinal coefficients

### utils
- `prepare_data()` - Separate features/target, encode labels
- `create_results_directory()` - Organized output folders
- `save_results()` - Save to JSON/pickle

## 🔧 Configuration

**Models:**  
`max_depth=3` (small datasets), `n_estimators=100`, `k=15-20` features → narrow to top 3-5

**Stats:**  
`n_permutations=1000`, p < 0.05 = significant

**Viz:**  
Log scale for wide ranges, dpi=300 for publication

## 📊 Output Structure

`create_results_directory()` organizes outputs:

```
results/your_analysis/
├── figures/              # All plots
├── data/                 # CSVs, feature lists
├── models/               # Saved models
└── results.json          # Summary metrics
```

## 🛠️ No Install Usage

```python
import sys
sys.path.insert(0, 'src')
from preprocessing.data_processing import load_and_impute
```

Recommended: Use `pip install -e .` for clean imports.

## 🔍 Troubleshooting

```bash
# Import errors
pip list | grep ml-analysis
pip install -e . --force-reinstall

# Optional dependencies
pip install umap-learn     # For UMAP
pip install mord           # For ordinal regression
```

## 📝 Dependencies

**Core:** pandas, numpy, scikit-learn, matplotlib, seaborn, shap  
**Optional:** umap-learn, mord

## 📄 License

[Your License]

## 🤝 Contributing

Format code with `black` before submitting.

---

**Version:** 0.1.0 | **Python:** ≥3.8 | **See also:** `ORDINAL_INTEGRATION.md` for ordinal regression details