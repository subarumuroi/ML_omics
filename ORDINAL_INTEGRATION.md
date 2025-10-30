# Ordinal Regression Guide

## When to Use

**Ordinal Regression** - Target has natural ordering (Green < Ripe < Overripe)
- ✅ Respects ordering in predictions
- ✅ Provides interpretable coefficients (direction of effect)
- ✅ Reports MAE (how many categories off)
- ✅ Auto log-transform and scaling

**Random Forest** - Complex non-linear patterns
- ✅ No ordering assumption needed
- ✅ Captures feature interactions
- ✅ Robust feature importance
- ✅ Permutation testing, SHAP analysis

**Use Both** - Compare approaches and validate results

## Quick Start

```bash
pip install -e ".[ordinal]"
```

```python
from preprocessing import load_and_impute
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from utils import prepare_data

# Load and encode
df = load_and_impute("data.csv", group_col='Groups', impute=True)
X, y_raw, _, _, _ = prepare_data(df)
y = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])

# Train with CV
results = train_evaluate_ordinal(X, y)

print(f"Accuracy: {results['mean_accuracy']:.3f}")
print(f"MAE: {results['mean_mae']:.3f}")  # Mean categories off
```

## Key Improvements from Original Notebook

**Before:**
```python
# Manual preprocessing
X_log = np.log10(features)
X = MinMaxScaler().fit_transform(X_log)

# Manual encoding
y = target.map({'Green': 0, 'Ripe': 1, 'Overripe': 2}).values

# Fit without CV
model = LogisticAT()
model.fit(X, y)
coefficients = model.coef_
```

**After:**
```python
# Automatic preprocessing + CV
results = train_evaluate_ordinal(
    X, y,
    log_transform=True,
    scaler='minmax'
)

# Formatted coefficients
coef_df = results['coefficients']  # Sorted DataFrame
```

## Visualization

```python
from visualization.ordinal_plots import (
    plot_ordinal_coefficients,
    plot_ordinal_confusion_matrix,
    plot_ordinal_cv_performance,
)

# Ordinal-specific
plot_ordinal_coefficients(results['coefficients'], top_n=10)
plot_ordinal_confusion_matrix(results['all_true'], results['all_pred'])

# Reuse existing plots
from visualization import plot_compound_trends
top_features = results['coefficients'].head(10)['Feature'].tolist()
plot_compound_trends(df, top_features, scale='log')
```

## Interoperability

### Use ordinal coefficients with RF
```python
# Get top features from ordinal
ordinal_results = train_evaluate_ordinal(X, y_ordinal)
top_features = ordinal_results['coefficients'].head(10)['Feature'].tolist()

# Train RF on subset
X_subset = X[top_features]
rf_clf = train_evaluate_model(X_subset, y_encoded)
```

### Compare both approaches
```python
from models.ordinal import compare_ordinal_models

# Compare ordinal model types
comparison = compare_ordinal_models(X, y, 
    model_types=['LogisticAT', 'LogisticIT', 'LogisticSE'])

# Compare with RF
rf_results = train_evaluate_model(X, y_encoded)
print(f"RF: {rf_results.cv_results['mean_accuracy']:.3f}")
print(f"Ordinal: {ordinal_results['mean_accuracy']:.3f} (MAE: {ordinal_results['mean_mae']:.3f})")
```

## Configuration

```python
results = train_evaluate_ordinal(
    X, y,
    model_type='LogisticAT',    # 'LogisticAT', 'LogisticIT', 'LogisticSE'
    scaler='minmax',             # or 'standard'
    log_transform=True,          # Apply log10
    n_splits=None,               # Auto-detect from min class size
    random_state=42
)
```

## Complete Example

See `examples/ordinal_workflow.py` for full pipeline including:
- Data loading, ordinal encoding
- Model training with CV
- All visualizations (coefficients, confusion, trends, correlations, violins)
- Model comparison
- Automated result saving

```bash
python examples/ordinal_workflow.py
```

## New Functions Added

**From your notebook, now available for any model:**
- `plot_correlation_heatmap()` - Correlation matrix
- `plot_violin_by_group()` - Violin plots by group

Both in `visualization.data_plots` module.

## Best Practices

1. **Always use CV** - `train_evaluate_ordinal()` includes it
2. **Report MAE** - Shows average category error
3. **Compare with RF** - Validate with different approach
4. **Visualize coefficients** - Positive = increases ripeness

---

**Resources:** [mord docs](https://pythonhosted.org/mord/) | `examples/ordinal_workflow.py`