# Quick Reference Card

## 🚀 Getting Started (2 Minutes)

```bash
# 1. Navigate to package directory
cd /path/to/ml-analysis-package

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# 4. Install package
pip install -e ".[all]"

# 5. Test it works
python -c "from preprocessing import load_and_impute; print('✅ Ready!')"
```

## 📥 Installation Commands

```bash
# Basic install
pip install -e .

# With UMAP support
pip install -e ".[umap]"

# With dev tools (testing, linting)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"

# From requirements.txt (alternative)
pip install -r requirements.txt
```

## 🔌 Import Examples

```python
# === Preprocessing ===
from preprocessing import (
    load_and_impute,              # Load CSV with imputation
    select_k_best_features,       # Select top K features
    get_top_n_features,           # Get top N feature names
)

# === Models ===
from models import (
    train_evaluate_model,         # Quick train + CV
    RFClassifierCV,               # RF with CV support
)
from models.evaluation import (
    compare_models,               # Compare model results
    create_results_summary,       # Summarize results
)

# === Analysis ===
from analysis import (
    get_feature_importance_df,    # Get importance DataFrame
    compute_permutation_importance, # Permutation importance
    permutation_test,             # Statistical significance test
    get_shap_values,              # SHAP explainability
)

# === Visualization ===
from visualization import (
    plot_confusion_matrix,        # Confusion matrix
    plot_feature_importance,      # Importance bar chart
    plot_compound_trends,         # Line plots across groups
    plot_compound_boxplots,       # Distribution boxplots
    plot_tsne,                    # t-SNE visualization
    plot_umap,                    # UMAP visualization
    plot_shap_summary,            # SHAP summary plots
)

# === Utils ===
from utils import (
    prepare_data,                 # Separate X, y and encode
    print_data_summary,           # Print dataset stats
    save_results,                 # Save to JSON/pickle
    create_results_directory,     # Create organized folders
)
```

## 🎯 Common Workflows

### Minimal Workflow
```python
from preprocessing import load_and_impute, select_k_best_features
from models import train_evaluate_model
from utils import prepare_data

# Load and prepare
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, _, y, le, _ = prepare_data(df)

# Select and train
X_sel, features, _ = select_k_best_features(X, y, k=15)
clf = train_evaluate_model(X_sel, y)

print(f"Accuracy: {clf.cv_results['mean_accuracy']:.3f}")
```

### Full Analysis
```python
from preprocessing import load_and_impute, select_k_best_features, get_top_n_features
from models import train_evaluate_model
from analysis import permutation_test, get_feature_importance_df
from visualization import plot_confusion_matrix, plot_compound_trends
from utils import prepare_data, create_results_directory, save_results
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Setup
dirs = create_results_directory('results', 'exp1')

# Load data
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, _, y, le, _ = prepare_data(df)

# Select features and train
X_sel, features, _ = select_k_best_features(X, y, k=15)
clf = train_evaluate_model(X_sel, y)

# Importance and visualization
importance_df = get_feature_importance_df(clf.model, features)
top_5 = get_top_n_features(importance_df, n=5)

plot_confusion_matrix(clf.confusion_matrix, le.classes_)
plot_compound_trends(df, top_5, scale='log')

# Permutation test
cv = StratifiedKFold(n_splits=4, shuffle=False)
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
perm_results = permutation_test(rf, X_sel, y, cv, n_permutations=1000)

# Save everything
save_results({
    'accuracy': clf.cv_results['mean_accuracy'],
    'p_value': perm_results['p_value'],
    'top_features': top_5
}, dirs['base'] / 'results.json')
```

## 🛠️ Development Commands

```bash
# Format code
black src/

# Check linting
flake8 src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Type checking
mypy src/

# Build distribution
python -m build

# Install from built distribution
pip install dist/ml_analysis_package-0.1.0-py3-none-any.whl
```

## 📊 Key Functions Quick Reference

| Task | Function | Module |
|------|----------|--------|
| Load data with imputation | `load_and_impute()` | preprocessing |
| Select top K features | `select_k_best_features()` | preprocessing |
| Prepare X, y | `prepare_data()` | utils |
| Train model with CV | `train_evaluate_model()` | models |
| Get feature importance | `get_feature_importance_df()` | analysis |
| Permutation test | `permutation_test()` | analysis |
| Plot confusion matrix | `plot_confusion_matrix()` | visualization |
| Plot feature trends | `plot_compound_trends()` | visualization |
| Dimensionality reduction | `plot_tsne()`, `plot_umap()` | visualization |
| Save results | `save_results()` | utils |

## 🔍 Troubleshooting

```bash
# Can't import modules?
pip list | grep ml-analysis  # Check if installed
pip install -e .            # Reinstall

# Import working but changes not reflected?
# You're in editable mode, changes should work immediately
# If not, try:
pip install -e . --force-reinstall

# UMAP not found?
pip install umap-learn

# SHAP installation issues?
pip install shap --no-cache-dir

# Virtual environment not activated?
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

## 📁 Project Structure at a Glance

```
ml-analysis-package/
├── pyproject.toml          # Package config
├── src/                    # Source code
│   ├── preprocessing/      # Data & features
│   ├── models/             # ML models
│   ├── analysis/           # Statistics
│   ├── visualization/      # Plots
│   └── utils/              # Helpers
└── examples/               # Usage examples
```

## 💡 Pro Tips

1. **Always activate venv first**: `source venv/bin/activate`
2. **Use editable install**: `pip install -e .` for development
3. **Save results**: Use `create_results_directory()` for organization
4. **Permutation tests**: 1000 iterations for reliable p-values
5. **Feature selection**: Start with k=15, narrow to top 3-5
6. **Visualization**: Use log scale for wide-ranging data
7. **SHAP**: Run on selected features, not all features

## 🔗 Useful Links

- Package repo: `https://github.com/yourusername/ml-analysis-package`
- Issue tracker: `https://github.com/yourusername/ml-analysis-package/issues`
- Full docs: `README.md`, `INSTALLATION.md`, `PACKAGE_STRUCTURE.md`

## 📝 License

[Your License] - See LICENSE file

---

**Last Updated**: 2024
**Version**: 0.1.0