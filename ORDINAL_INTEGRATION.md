# Ordinal Regression Integration Guide

## Overview

Ordinal regression has been integrated into the package for analyzing **ordered categorical outcomes** (e.g., Green < Ripe < Overripe). This is more appropriate than standard classification when your target variable has a natural ordering.

## 🎯 When to Use Ordinal vs Random Forest

### Use **Ordinal Regression** when:
- ✅ Target has natural ordering (e.g., low/medium/high, green/ripe/overripe)
- ✅ You want to respect the ordering in predictions
- ✅ You care about "how far off" predictions are (MAE matters)
- ✅ You want interpretable coefficients showing direction of effect
- ✅ You want to apply log transformation and scaling automatically

### Use **Random Forest** when:
- ✅ Target classes have no meaningful order
- ✅ You want complex non-linear interactions
- ✅ You want robust feature importance
- ✅ You don't need coefficient interpretation
- ✅ You want permutation testing and SHAP analysis

### Use **Both** when:
- ✅ You want to compare approaches
- ✅ You want cross-validation of results
- ✅ You want both interpretability (ordinal) and robustness (RF)

## 📦 Installation

```bash
# Install with ordinal support
pip install -e ".[ordinal]"

# Or install everything
pip install -e ".[all]"
```

## 🚀 Quick Start

### Basic Usage

```python
from preprocessing import load_and_impute
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from utils import prepare_data

# Load data
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, y_raw, _, _, _ = prepare_data(df)

# Encode as ordinal
y = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])

# Train with cross-validation
results = train_evaluate_ordinal(X, y)

print(f"Accuracy: {results['mean_accuracy']:.3f}")
print(f"MAE: {results['mean_mae']:.3f}")  # Mean Absolute Error (ordinal-specific)
```

## 📊 Key Differences from Your Original Code

### Your Original Code:
```python
# Manual preprocessing
X_log = np.log10(features)
scaler = MinMaxScaler()
X = scaler.fit_transform(X_log)

# Manual encoding
ordinal_map = {'Green': 0, 'Ripe': 1, 'Overripe': 2}
y = target.map(ordinal_map).values

# Fit model
model = LogisticAT()
model.fit(X, y)

# Get coefficients manually
coefficients = model.coef_
```

### Package Version (Integrated):
```python
# Preprocessing happens automatically
results = train_evaluate_ordinal(
    X, y,
    log_transform=True,      # Automatically applies log10
    scaler='minmax',         # Automatically scales
    model_type='LogisticAT'  # Same model
)

# Coefficients already formatted as DataFrame
coef_df = results['coefficients']  # Sorted by importance
```

## 🔄 Integration with Existing Workflow

### Compare Random Forest and Ordinal

```python
from preprocessing import load_and_impute, select_k_best_features
from models import train_evaluate_model
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from utils import prepare_data

# Load data
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, y_raw, _, le, _ = prepare_data(df)

# For Random Forest (standard encoding)
X_sel, features, _ = select_k_best_features(X, y_raw, k=15)
rf_clf = train_evaluate_model(X_sel, le.transform(y_raw))

# For Ordinal Regression (ordinal encoding)
y_ordinal = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])
ordinal_results = train_evaluate_ordinal(X, y_ordinal)

# Compare
print("Random Forest:")
print(f"  Accuracy: {rf_clf.cv_results['mean_accuracy']:.3f}")
print(f"\nOrdinal Regression:")
print(f"  Accuracy: {ordinal_results['mean_accuracy']:.3f}")
print(f"  MAE: {ordinal_results['mean_mae']:.3f}")
```

## 📈 Visualization Functions

### Ordinal-Specific Plots

```python
from visualization.ordinal_plots import (
    plot_ordinal_coefficients,
    plot_ordinal_confusion_matrix,
    plot_ordinal_predictions_vs_true,
    plot_ordinal_cv_performance,
)

# Coefficient plot (shows direction of effect)
plot_ordinal_coefficients(results['coefficients'], top_n=10)

# Confusion matrix
plot_ordinal_confusion_matrix(results['all_true'], results['all_pred'])

# Predictions vs true (diagonal = perfect)
plot_ordinal_predictions_vs_true(results['all_true'], results['all_pred'])

# CV performance
plot_ordinal_cv_performance(results['cv_scores'], results['cv_mae'])
```

### Reuse Existing Visualizations

All existing visualization functions work with ordinal models:

```python
from visualization import plot_compound_trends, plot_compound_boxplots

# Get top features from ordinal coefficients
top_features = results['coefficients'].head(10)['Feature'].tolist()

# Use standard visualization functions
plot_compound_trends(df, top_features, scale='log')
plot_compound_boxplots(df, top_features)
```

## 🔧 Model Configuration

### Available Ordinal Models

```python
from models.ordinal import compare_ordinal_models

# Compare different ordinal model types
comparison = compare_ordinal_models(
    X, y,
    model_types=['LogisticAT', 'LogisticIT', 'LogisticSE']
)

print(comparison)
```

**Model Types:**
- `LogisticAT`: All-Threshold (most common, your original model)
- `LogisticIT`: Immediate-Threshold  
- `LogisticSE`: Squared-Error

### Preprocessing Options

```python
results = train_evaluate_ordinal(
    X, y,
    model_type='LogisticAT',
    scaler='minmax',          # or 'standard'
    log_transform=True,       # Apply log10 transformation
    n_splits=4,               # CV folds
    random_state=42
)
```

## 📁 Complete Example

See `examples/ordinal_workflow.py` for a comprehensive example that includes:
- ✅ Data loading and preprocessing
- ✅ Ordinal model training with CV
- ✅ Coefficient visualization
- ✅ Confusion matrix and predictions
- ✅ Compound trend analysis
- ✅ Correlation heatmap
- ✅ Violin plots
- ✅ Model comparison (multiple ordinal types)
- ✅ Comparison with Random Forest
- ✅ Automated result saving

Run it:
```bash
cd examples
python ordinal_workflow.py
```

## 🎨 New Visualization Functions Added

### From Your Original Code

These have been integrated into `visualization/data_plots.py`:

1. **`plot_correlation_heatmap()`** - Correlation matrix of features
2. **`plot_violin_by_group()`** - Violin plots across groups

Both are now available for use with **any model** (RF or ordinal):

```python
from visualization.data_plots import (
    plot_correlation_heatmap,
    plot_violin_by_group
)

# Works with any feature set
plot_correlation_heatmap(df, top_features)
plot_violin_by_group(df, top_features, group_col='Groups')
```

## 🔑 Key Benefits of Integration

### Before (Your Notebook):
- ❌ Manual preprocessing (log, scale)
- ❌ No cross-validation
- ❌ Manual coefficient extraction
- ❌ Hardcoded top N features
- ❌ Repeated code for visualizations
- ❌ No comparison with other methods

### After (Package):
- ✅ Automatic preprocessing (configurable)
- ✅ Built-in cross-validation with MAE
- ✅ Formatted coefficient DataFrame
- ✅ Flexible N features parameter
- ✅ Reusable visualization functions
- ✅ Easy comparison with RF and other models
- ✅ Integrated into full workflow

## 💡 Best Practices

### 1. Always Use Cross-Validation
```python
# Good
results = train_evaluate_ordinal(X, y)  # Includes CV

# Not recommended for final results
clf = OrdinalClassifier()
clf.fit(X, y)  # No CV
```

### 2. Report Both Accuracy and MAE
```python
print(f"Accuracy: {results['mean_accuracy']:.3f}")
print(f"MAE: {results['mean_mae']:.3f}")  # Ordinal-specific metric
```

MAE tells you **how many categories off** your predictions are on average.

### 3. Compare with Random Forest
Since both are available, compare approaches:
```python
# Ordinal respects ordering
ordinal_results = train_evaluate_ordinal(X, y_ordinal)

# RF doesn't assume ordering but captures complex interactions
rf_clf = train_evaluate_model(X_sel, y_encoded)
```

### 4. Visualize Coefficients
Ordinal coefficients show **direction of effect**:
- Positive coefficient → increases ripeness
- Negative coefficient → decreases ripeness

```python
plot_ordinal_coefficients(results['coefficients'], top_n=10)
```

## 🔄 Interoperability Examples

### Use Ordinal Coefficients for RF Feature Selection
```python
# Get top features from ordinal model
ordinal_results = train_evaluate_ordinal(X, y_ordinal)
top_features = ordinal_results['coefficients'].head(10)['Feature'].tolist()

# Train RF on those features
X_subset = X[top_features]
rf_clf = train_evaluate_model(X_subset, y_encoded)
```

### Use RF Importance for Ordinal Model
```python
# Get top features from RF
rf_clf = train_evaluate_model(X, y)
from analysis import get_feature_importance_df
importance_df = get_feature_importance_df(rf_clf.model, X.columns)
top_features = importance_df.head(10)['Feature'].tolist()

# Train ordinal model on those features
X_subset = X[top_features]
ordinal_results = train_evaluate_ordinal(X_subset, y_ordinal)
```

## 📚 Additional Resources

- **mord documentation**: [https://pythonhosted.org/mord/](https://pythonhosted.org/mord/)
- **Ordinal regression theory**: Frank & Hall (2001)
- **Package examples**: `examples/ordinal_workflow.py`

## 🤝 Summary

Ordinal regression is now **fully integrated** and **interoperable** with the rest of the package:

- ✅ Same data loading and preprocessing utilities
- ✅ Compatible with all visualization functions
- ✅ Cross-validation built-in
- ✅ Easy comparison with Random Forest
- ✅ Automatic log transformation and scaling
- ✅ Ordinal-specific metrics (MAE)
- ✅ Works with the existing workflow

You can use ordinal regression as a **drop-in alternative** to Random Forest when your outcome has natural ordering!