"""
Data preparation and manipulation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def prepare_data(df, target_col='Groups', drop_missing=True, verbose=True):
    """
    Prepare data by separating features and target, handling missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str, default='Groups'
        Target column name
    drop_missing : bool, default=True
        Drop columns with missing values
    verbose : bool, default=True
        Print dropped columns
        
    Returns
    -------
    tuple
        (X, y_raw, y_encoded, label_encoder, dropped_cols)
    """
    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col])
    
    dropped_cols = []
    if drop_missing:
        missing_cols = X_raw.columns[X_raw.isna().any()].tolist()
        dropped_cols = missing_cols
        
        if missing_cols and verbose:
            print(f"\nDropped {len(missing_cols)} columns with missing values")
        
        X = X_raw.drop(columns=missing_cols)
    else:
        X = X_raw
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    if verbose:
        print(f"Classes: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    
    return X, y_raw, y_encoded, le, dropped_cols


def set_categorical_order(df, col, categories, ordered=True):
    """
    Set categorical order for a column.
    
    Useful for ensuring proper ordering in plots (e.g., 'Green', 'Ripe', 'Overripe').
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column name to convert to categorical
    categories : list
        Ordered list of categories
    ordered : bool, default=True
        Whether categories are ordered
        
    Returns
    -------
    pd.DataFrame
        DataFrame with updated categorical column (modified in place)
    """
    df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)
    return df


def get_cv_splits(y, n_splits=None, min_splits=2):
    """
    Get appropriate number of CV splits based on class distribution.
    
    Parameters
    ----------
    y : np.ndarray
        Target labels (encoded)
    n_splits : int, optional
        Number of splits. If None, uses minimum class count
    min_splits : int, default=2
        Minimum number of splits
        
    Returns
    -------
    int
        Number of splits to use
    """
    if n_splits is None:
        n_splits = max(np.bincount(y).min(), min_splits)
    return n_splits


def print_data_summary(X, y, label_encoder=None):
    """
    Print summary statistics of the dataset.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target labels (encoded)
    label_encoder : sklearn.LabelEncoder, optional
        Label encoder for class names
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for u, c in zip(unique, counts):
        if label_encoder is not None:
            label = label_encoder.inverse_transform([u])[0]
        else:
            label = u
        print(f"  {label}: {c} samples ({c/len(y)*100:.1f}%)")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean values range: [{X.mean().min():.2e}, {X.mean().max():.2e}]")
    print(f"  Std values range: [{X.std().min():.2e}, {X.std().max():.2e}]")
    
    print("="*60 + "\n")


def check_class_balance(y, label_encoder=None, threshold=0.3):
    """
    Check if classes are balanced and print warning if not.
    
    Parameters
    ----------
    y : np.ndarray
        Target labels (encoded)
    label_encoder : sklearn.LabelEncoder, optional
        Label encoder for class names
    threshold : float, default=0.3
        Threshold for imbalance warning (ratio of smallest to largest class)
        
    Returns
    -------
    dict
        Dictionary with class counts and balance ratio
    """
    unique, counts = np.unique(y, return_counts=True)
    
    min_count = counts.min()
    max_count = counts.max()
    balance_ratio = min_count / max_count
    
    if balance_ratio < threshold:
        print(f"⚠️  WARNING: Classes are imbalanced (ratio: {balance_ratio:.2f})")
        print(f"   Consider using stratified sampling or class weights.")
    
    result = {'counts': dict(zip(unique, counts)), 'balance_ratio': balance_ratio}
    return result


def drop_index_columns(df):
    """
    Drop unnamed index columns from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with index columns removed
    """
    index_cols = [col for col in df.columns if col.lower().startswith('unnamed')]
    if index_cols:
        df = df.drop(columns=index_cols)
        print(f"Dropped index columns: {index_cols}")
    return df


def remove_low_variance_features(X, threshold=0.01, verbose=True):
    """
    Remove features with variance below threshold.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    threshold : float, default=0.01
        Variance threshold
    verbose : bool, default=True
        Print removed features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with low variance features removed
    """
    variances = X.var()
    low_var_cols = variances[variances < threshold].index.tolist()
    
    if low_var_cols and verbose:
        print(f"\nRemoved {len(low_var_cols)} low variance features:")
        for col in low_var_cols:
            print(f"  - {col} (var: {variances[col]:.2e})")
    
    return X.drop(columns=low_var_cols)