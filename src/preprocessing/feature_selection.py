"""
Feature selection and importance analysis module.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance


def select_k_best_features(X, y, k=15, score_func=f_classif):
    """
    Select top k features using univariate statistical tests.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target labels
    k : int, default=15
        Number of features to select
    score_func : callable, default=f_classif
        Scoring function for feature selection
        
    Returns
    -------
    tuple
        (X_selected, selected_feature_names, selector)
    """
    k = min(k, X.shape[1])  # Don't select more features than available
    
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    print(f"Selected {k} features from {X.shape[1]} total features")
    
    return X_selected, selected_features, selector


def get_feature_importance_df(model, feature_names, top_n=None, sort=True):
    """
    Get feature importances as a sorted DataFrame.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list or pd.Index
        Names of features
    top_n : int, optional
        Return only top N features
    sort : bool, default=True
        Sort by importance (descending)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Feature and Importance columns
    """
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    if sort:
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    return importance_df


def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42,
                                   scoring='accuracy'):
    """
    Compute permutation importance for robustness check.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    n_repeats : int, default=10
        Number of permutations
    random_state : int, default=42
        Random state
    scoring : str, default='accuracy'
        Scoring metric
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Feature, Importance, and Std columns
    """
    perm_result = permutation_importance(
        model, X, y, 
        scoring=scoring, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    
    feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
    
    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_result.importances_mean,
        'Std': perm_result.importances_std,
    }).sort_values(by='Importance', ascending=False)
    
    return perm_df, perm_result


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
    return X[feature_names]