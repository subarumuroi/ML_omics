"""
Utility functions for data preparation and analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def prepare_data(df, target_col='Groups', drop_missing=True):
    """
    Prepare data by separating features and target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str, default='Groups'
        Name of target column
    drop_missing : bool, default=True
        Drop columns with missing values
        
    Returns
    -------
    tuple
        (X, y, y_encoded, label_encoder, dropped_cols)
    """
    # Separate features and target
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])
    
    dropped_cols = []
    if drop_missing:
        # Identify columns with missing values
        missing_cols = X_raw.columns[X_raw.isna().any()].tolist()
        dropped_cols = missing_cols
        
        if missing_cols:
            print(f"\nDropped {len(missing_cols)} columns due to missing values:")
            for col in missing_cols:
                print(f" - {col}")
        
        X = X_raw.drop(columns=missing_cols)
    else:
        X = X_raw
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    return X, y_raw, y_encoded, le, dropped_cols


def set_categorical_order(df, col, categories, ordered=True):
    """
    Set categorical order for a column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column name
    categories : list
        Ordered list of categories
    ordered : bool, default=True
        Whether categories are ordered
        
    Returns
    -------
    pd.DataFrame
        DataFrame with updated categorical column
    """
    df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)
    return df


def get_cv_splits(y, n_splits=None):
    """
    Get appropriate number of CV splits based on class distribution.
    
    Parameters
    ----------
    y : np.ndarray
        Target labels
    n_splits : int, optional
        Number of splits (if None, uses minimum class count)
        
    Returns
    -------
    int
        Number of splits to use
    """
    if n_splits is None:
        n_splits = np.bincount(y).min()
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
        label = label_encoder.inverse_transform([u])[0] if label_encoder else u
        print(f"  {label}: {c} samples")
    
    print("="*60 + "\n")


def create_results_summary(cv_results, perm_test_results=None):
    """
    Create summary dictionary of model results.
    
    Parameters
    ----------
    cv_results : dict
        Cross-validation results from model
    perm_test_results : dict, optional
        Permutation test results
        
    Returns
    -------
    dict
        Summary of all results
    """
    summary = {
        'cv_accuracy': cv_results['mean_accuracy'],
        'cv_accuracy_std': cv_results['std_accuracy'],
        'cv_balanced_accuracy': cv_results['mean_balanced_accuracy'],
        'cv_balanced_accuracy_std': cv_results['std_balanced_accuracy'],
    }
    
    if perm_test_results is not None:
        summary.update({
            'permutation_p_value': perm_test_results['p_value'],
            'permutation_mean': perm_test_results['perm_mean'],
            'permutation_std': perm_test_results['perm_std'],
        })
    
    return summary


def save_results(results_dict, filepath):
    """
    Save results dictionary to file.
    
    Parameters
    ----------
    results_dict : dict
        Results to save
    filepath : str
        Output filepath (supports .json, .pkl)
    """
    import json
    import pickle
    from pathlib import Path
    
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for key, val in results_dict.items():
            if isinstance(val, np.ndarray):
                serializable[key] = val.tolist()
            elif isinstance(val, (np.integer, np.floating)):
                serializable[key] = float(val)
            else:
                serializable[key] = val
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results_dict, f)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    print(f"Results saved to: {filepath}")