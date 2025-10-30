# Data Preprocessing Guide

## Your Banana Dataset Workflow

### Preprocessing Strategies

Your `load_and_impute()` function supports multiple strategies for handling missing mass spectrometry data:

```python
from preprocessing import load_and_impute

# Strategy 1: Impute by group median + fill remaining with constant
df = load_and_impute('data/badata.csv', group_col='Groups', fill_value=50, impute=True)
# → Equivalent to: badata_imputed_filled.csv

# Strategy 2: Impute by group median only
df = load_and_impute('data/badata.csv', group_col='Groups', fill_value=None, impute=True)
# → Equivalent to: badata_imputed.csv (may have NaNs)

# Strategy 3: Fill all missing with constant
df = load_and_impute('data/badata.csv', group_col='Groups', fill_value=50, impute=False)
# → Equivalent to: badata_filled.csv
```

### How It Works

1. **Median Imputation by Group** (`impute=True`)
   - Groups data by `group_col` (e.g., 'Groups')
   - For each group, fills missing values with group median
   - Handles case where entire group has missing values

2. **Constant Fill** (`fill_value=50`)
   - Fills any remaining NaNs with constant
   - Useful for mass spec data (undetectable = low abundance)
   - Converts integer columns to float (normal behavior)

### Complete Example

```python
from preprocessing import load_and_impute
from utils import prepare_data, set_categorical_order

# Load with your preferred strategy
df = load_and_impute(
    file_path='data/badata.csv',
    group_col='Groups',
    fill_value=50,      # Small constant for undetectable values
    impute=True         # Use group median first
)

# Optional: Drop unnamed index column
if df.columns[0].startswith('Unnamed'):
    df = df.drop(columns=[df.columns[0]])

# Set ordinal ordering
df = set_categorical_order(df, 'Groups', ['Green', 'Ripe', 'Overripe'])

# Prepare for modeling
X, y_raw, y, le, dropped = prepare_data(df, target_col='Groups')

# Now X is ready for any model!
```

## ⚠️ Important Notes

### 1. Index Handling

Your function uses `grouped.apply()` which can create MultiIndex. The current implementation handles this, but if you see issues:

```python
# After load_and_impute, check for MultiIndex
if isinstance(df.index, pd.MultiIndex):
    df = df.reset_index(drop=True)
```

### 2. Unnamed Index Columns

Your CSV may have an unnamed index column. Handle it:

```python
# Check first column
if df.columns[0].startswith('Unnamed') or df.columns[0].startswith('index'):
    df = df.drop(columns=[df.columns[0]])
```

### 3. Data Type Conversion

When filling NaNs with a constant:
- Integer columns → Float columns (normal pandas behavior)
- This doesn't affect modeling

### 4. Missing Values After Imputation

If using `impute=True` without `fill_value`, some NaNs may remain:
- Entire group has missing values for a column
- Use `prepare_data(drop_missing=True)` to handle these

## Verification

Test your preprocessing:

```python
df = load_and_impute('data/badata.csv', 'Groups', fill_value=50, impute=True)

print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")  # Should be 0
print(f"Groups: {df['Groups'].unique()}")
print(f"Index type: {type(df.index)}")  # Should be RangeIndex
```

Expected output:
```
Shape: (12, N)  # 12 samples (4 per group), N features
Missing values: 0
Groups: ['Green' 'Ripe' 'Overripe']
Index type: <class 'pandas.core.indexes.range.RangeIndex'>
```

## Integration with Models

### Random Forest

```python
from models import train_evaluate_model
from preprocessing import select_k_best_features

df = load_and_impute('data/badata.csv', 'Groups', fill_value=50, impute=True)
X, _, y, le, _ = prepare_data(df)

X_selected, features, _ = select_k_best_features(X, y, k=15)
clf = train_evaluate_model(X_selected, y)
```

### Ordinal Regression

```python
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target

df = load_and_impute('data/badata.csv', 'Groups', fill_value=50, impute=True)
X, y_raw, _, _, _ = prepare_data(df)

y_ordinal = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])
results = train_evaluate_ordinal(X, y_ordinal)
```

## Testing

Run the verification script:

```bash
python test_banana_workflow.py
```

This tests:
- ✓ All preprocessing strategies
- ✓ Full RF workflow
- ✓ Full ordinal workflow
- ✓ Visualizations

## Common Issues & Solutions

### Issue 1: MultiIndex after load_and_impute
```python
df = df.reset_index(drop=True)
```

### Issue 2: "Groups column not found"
```python
# Check column names
print(df.columns.tolist())

# Groups column might be named differently
df = load_and_impute('data/badata.csv', group_col='Groups')  # Match your CSV
```

### Issue 3: Too many NaNs after imputation
```python
# Use fill_value to handle remaining NaNs
df = load_and_impute('data/badata.csv', 'Groups', fill_value=50, impute=True)
```

### Issue 4: prepare_data() fails
```python
# Drop columns with missing values
X, y_raw, y, le, dropped = prepare_data(df, drop_missing=True)
print(f"Dropped {len(dropped)} columns")
```

## Best Practice Workflow

```python
# 1. Load with both strategies
df = load_and_impute('data/badata.csv', 'Groups', fill_value=50, impute=True)

# 2. Clean up
if df.columns[0].startswith('Unnamed'):
    df = df.drop(columns=[df.columns[0]])

# 3. Set ordering (for ordinal models)
df = set_categorical_order(df, 'Groups', ['Green', 'Ripe', 'Overripe'])

# 4. Prepare
X, y_raw, y, le, _ = prepare_data(df, drop_missing=True)

# 5. Ready to use!
```

## Summary

✅ Your `load_and_impute()` function is **fully compatible**  
✅ All preprocessing strategies work  
✅ Both RF and ordinal models supported  
✅ Handles missing data appropriately for mass spec  

The package is designed to work seamlessly with your banana dataset workflow!