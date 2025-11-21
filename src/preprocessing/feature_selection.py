"""
Feature selection methods for dimensionality reduction.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def select_k_best_features(X, y, k=15, score_func=f_classif, verbose=True):
    """
    Select top k features using univariate statistical tests.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray or pd.Series
        Target labels
    k : int, default=15
        Number of features to select
    score_func : callable, default=f_classif
        Scoring function (f_classif, chi2, mutual_info_classif)
    verbose : bool, default=True
        Print selection summary
        
    Returns
    -------
    tuple
        (X_selected, selected_feature_names, selector)
        X_selected and selected_feature_names are sorted by score (descending)
    """
    # Convert y to array if Series
    if isinstance(y, pd.Series):
        y = y.values
    
    k = min(k, X.shape[1])
    
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected features and their scores
    feature_mask = selector.get_support()
    selected_features = X.columns[feature_mask]
    selected_scores = selector.scores_[feature_mask]
    
    # Sort by score (descending)
    sorted_indices = np.argsort(selected_scores)[::-1]
    selected_features = selected_features[sorted_indices]
    X_selected = X_selected[:, sorted_indices]
    
    if verbose:
        print(f"Selected {k} features from {X.shape[1]} total features")
    
    return X_selected, selected_features, selector


def get_feature_scores(X, y, score_func=f_classif):
    """
    Get univariate scores for all features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target labels
    score_func : callable, default=f_classif
        Scoring function
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Feature and Score columns, sorted by score
    """
    selector = SelectKBest(score_func=score_func, k='all')
    selector.fit(X, y)
    
    scores_df = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_,
        'P_value': selector.pvalues_
    }).sort_values(by='Score', ascending=False)
    
    return scores_df


def prepare_feature_subset(X, feature_names):
    """
    Extract subset of features from DataFrame.
    
    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix
    feature_names : list
        Names of features to extract
        
    Returns
    -------
    pd.DataFrame
        Subset of features
    """
    # Ensure all requested features exist
    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Features not found in X: {missing}")
    
    return X[feature_names]


def get_top_n_features(importance_df, n=3):
    """
    Get top N feature names from importance DataFrame.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    n : int, default=3
        Number of top features to return
        
    Returns
    -------
    list
        List of top N feature names
    """
    return importance_df.head(n)['Feature'].tolist()