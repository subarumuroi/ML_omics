# ML Omics Analysis Package

A modular Python package for machine learning analysis of omics data with ordinal regression, Random Forest classification, automated feature selection, and comprehensive visualization.

**Built for:** Multi-omics classification tasks (metabolomics, proteomics, transcriptomics) with emphasis on ordered outcomes.

## Quick Install
```bash
cd /path/to/ML_omics

# Install in editable mode (recommended for development)
pip install -e ".[all]"

# Verify installation
python -c "from preprocessing import load_and_impute; print('Ready!')"
```

**Requirements:** Python ≥3.8

---

## Quick Start

### Ordinal Regression (Recommended for Ordered Outcomes)

For data with natural ordering (Green < Ripe < Overripe, Mild < Moderate < Severe):
```python
from preprocessing import load_and_impute, select_k_best_features
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from utils import prepare_data

# Load and prepare data
df = load_and_impute("data.csv", group_col='Groups', 
                     impute=True, fill_value=0, drop_threshold=0.35)
X, y_raw, _, _, _ = prepare_data(df, target_col='Groups')

# Encode ordinal target
y = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])

# Automatic feature selection (tests k=1,2,3,5,7,10,15)
X_opt, features, _ = select_k_best_features(X, y, k=5)  # or use optimal k

# Train with cross-validation
results = train_evaluate_ordinal(X_opt, y, log_transform=True)

print(f"Accuracy: {results['mean_accuracy']:.3f}")
print(f"MAE: {results['mean_mae']:.3f}")  # Mean absolute error (categories off)
```

### Random Forest (General Classification)

For any classification task or baseline comparison:
```python
from models import train_evaluate_model

# Same data preparation as above
X_selected, features, _ = select_k_best_features(X, y, k=15)

# Train with CV
clf = train_evaluate_model(X_selected, y, n_estimators=100, max_depth=3)

print(f"CV Accuracy: {clf.cv_results['mean_accuracy']:.3f}")
```

---

## Package Structure
```
ML_omics/
├── src/
│   ├── preprocessing/      # load_and_impute, feature selection
│   ├── models/             # RF, ordinal regression, evaluation
│   ├── analysis/           # Feature importance, permutation tests
│   ├── visualization/      # Plotting functions
│   └── utils/              # Data prep, I/O
│
└── examples/
    ├── ordinal_regression_workflow.py   # Main workflow (recommended for data with classifiable stages (I.e. disease progression))
    └── random_forest_workflow.py        # Alternative baseline (RF can help rank important features for mixed variable problems with categorical outcomes)
```

---

## Key Features

### Automatic Data Handling

**Handles common omics data issues:**
- ✅ Missing values (group median imputation)
- ✅ Negative values (metabolomics baseline correction)
- ✅ Low-quality features (automatic filtering)
```python
df = load_and_impute(
    file_path="data.csv",
    group_col='Groups',
    impute=True,           # Median imputation by group
    fill_value=0,          # Fill remaining NaNs
    drop_threshold=0.35    # Drop features >35% missing
)
```

**Smart feature filtering:**
- Drops features where entire groups are missing (>35% threshold)
- Handles negative values from normalization/correction
- Preserves relative differences for classification

### Automated Feature Selection

Finds optimal number of features via cross-validation:
```python
# Tests multiple k values automatically
feature_counts = [1, 2, 3, 5, 7, 10, 15]
for k in feature_counts:
    # Evaluates accuracy and MAE
    # Selects k with best performance
```

Common results: **2-5 features sufficient** for small omics datasets (n=9-12)

### Ordinal Regression Models

**Three model types available:**
- `LogisticAT` - All-Threshold model (most flexible, **recommended**)
- `LogisticIT` - Immediate-Threshold model
- `LogisticSE` - Stereotype model (assumes equal spacing)
```python
from models.ordinal import compare_ordinal_models

# Compare all three
comparison = compare_ordinal_models(X, y, 
    model_types=['LogisticAT', 'LogisticIT', 'LogisticSE'])
```

### Rich Visualizations
```python
from visualization.ordinal_plots import (
    plot_ordinal_coefficients,
    plot_ordinal_confusion_matrix,
    plot_ordinal_cv_performance
)
from visualization import plot_compound_trends, plot_compound_boxplots

# Ordinal-specific
plot_ordinal_coefficients(results['coefficients'], top_n=10)
plot_ordinal_confusion_matrix(results['all_true'], results['all_pred'])

# Data exploration
plot_compound_trends(df, top_features, group_col='Groups', scale='log')
plot_compound_boxplots(df, top_features, group_col='Groups')
```

---

## Complete Workflow Example

See `examples/ordinal_workflow.py` for full pipeline including:

1. **Data loading** with automatic handling of missing/negative values
2. **Automated feature selection** (finds optimal k)
3. **Ordinal model training** with cross-validation
4. **Model comparison** (LogisticAT vs IT vs SE vs Random Forest)
5. **Comprehensive visualization** (coefficients, confusion matrix, compound trends)
6. **Automated results saving** (figures, data, JSON summaries)
```bash
# Run complete analysis
python examples/ordinal_workflow.py

# Results saved to:
results/ordinal_analysis/
├── figures/          # All plots
├── data/             # CSVs, feature rankings
└── ordinal_results.json
```

---

## Output Structure

Organized results directory:
```
results/your_analysis_name/
├── figures/
│   ├── ordinal_coefficients.png
│   ├── ordinal_confusion_matrix.png
│   ├── compound_trends_linear.png
│   └── compound_trends_log.png
├── data/
│   ├── ordinal_coefficients.csv
│   ├── feature_selection_analysis.csv
│   └── top10_univariate_scores.csv
└── ordinal_results.json
```

---

## Common Use Cases

### 1. Quick Classification
```python
from preprocessing import load_and_impute, select_k_best_features
from models import train_evaluate_model
from utils import prepare_data

df = load_and_impute("data.csv", group_col='Groups', impute=True, fill_value=0)
X, _, y, _, _ = prepare_data(df, target_col='Groups')
X_sel, features, _ = select_k_best_features(X, y, k=10)
clf = train_evaluate_model(X_sel, y)
print(f"Accuracy: {clf.cv_results['mean_accuracy']:.3f}")
```

### 2. Find Top Biomarkers
```python
from preprocessing import get_feature_scores

# Get univariate F-statistics
scores_df = get_feature_scores(X, y)
print(scores_df.head(10))  # Top 10 features with p-values
```

### 3. Rigorous Validation
```python
from analysis import permutation_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=3)

perm_results = permutation_test(rf, X, y, cv, n_permutations=1000)
print(f"p-value: {perm_results['p_value']:.4f}")
```

---

## Core Modules

### preprocessing
- `load_and_impute()` - Load CSV with smart missing data handling
- `select_k_best_features()` - Univariate feature selection (sorted by score)
- `get_feature_scores()` - Get F-statistics and p-values

### models
- `train_evaluate_model()` - Random Forest with CV
- `train_evaluate_ordinal()` - Ordinal regression with CV
- `compare_ordinal_models()` - Compare multiple ordinal models
- `encode_ordinal_target()` - Convert categories to ordinal integers

### visualization
- `plot_ordinal_coefficients()` - Feature coefficients bar plot
- `plot_ordinal_confusion_matrix()` - Ordinal-specific confusion matrix
- `plot_compound_trends()` - Line plots across groups
- `plot_compound_boxplots()` - Distribution boxplots
- `plot_correlation_heatmap()` - Feature correlations
- `plot_violin_by_group()` - Violin plots by group

### analysis
- `get_feature_importance_df()` - Tree-based importance
- `permutation_test()` - Statistical significance testing
- `get_shap_values()` - Model interpretability (requires `shap`)

### utils
- `prepare_data()` - Separate features/target, encode labels
- `create_results_directory()` - Organized output folders
- `save_results()` - Save summaries to JSON

---

## Best Practices

### For Small Datasets (n < 20)
- ✅ Use ordinal regression (respects ordering, better with small n)
- ✅ Keep k small (2-5 features)
- ✅ Use all data for visualization (no train/test split)
- ✅ Report MAE alongside accuracy
- ✅ Use default minmax scaler and remove outlier (standard scaler may be too sensitive when data is not normally distributed)
- ⚠️ Perfect accuracy suggests overfitting—need external validation

### For Medium Datasets (n = 20-100)
- ✅ Compare ordinal vs Random Forest
- ✅ Use k=10-15 features
- ✅ Permutation testing for significance
- ✅ SHAP for interpretability

### For Metabolomics Data
- ✅ Use `drop_threshold=0.35` (drops features where 1+ group entirely missing (when 3 equal n groups exist))
- ✅ Set `fill_value=0` (undetected = absent)
- ✅ Workflow handles negative values automatically

---

## Installation Options
```bash
# Core package only
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Specific extras
pip install -e ".[ordinal]"    # + mord for ordinal regression
pip install -e ".[umap]"        # + umap-learn
pip install -e ".[dev]"         # + testing tools
```

**Core dependencies:** pandas, numpy, scikit-learn, matplotlib, seaborn, shap  
**Optional:** mord (ordinal), umap-learn (UMAP)

---

## Troubleshooting

**Import errors after installation:**
```bash
pip list | grep ml-analysis
pip install -e . --force-reinstall
```

**Ordinal regression not available:**
```bash
pip install mord
```

**Warnings about FutureWarning/DeprecationWarning:**
These are cosmetic and don't affect results. Will be addressed in future updates.

---

## Citation

If you use this package, please cite:
```
[TBA]
```

## License

MIT License - see LICENSE file

---

**Version:** 0.1.0  
**Maintained by:** Subaru Muroi
**Issues:** subarumuroi@github.com