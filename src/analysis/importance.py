"""
Feature importance analysis methods.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


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
                                   scoring='accuracy', verbose=False):
    """
    Compute permutation importance for robustness check.
    
    This method randomly shuffles each feature and measures the decrease in 
    model performance, providing a measure of feature importance that is less
    biased than tree-based importance for correlated features.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    n_repeats : int, default=10
        Number of times to permute each feature
    random_state : int, default=42
        Random state for reproducibility
    scoring : str, default='accuracy'
        Scoring metric (accuracy, balanced_accuracy, f1, etc.)
    verbose : bool, default=False
        Print progress
        
    Returns
    -------
    tuple
        (perm_df, perm_result)
        - perm_df: DataFrame with Feature, Importance, and Std columns
        - perm_result: Raw PermutationImportanceResult object
    """
    if verbose:
        print(f"Computing permutation importance with {n_repeats} repeats...")
    
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
    
    if verbose:
        print("Top 5 features by permutation importance:")
        print(perm_df.head())
    
    return perm_df, perm_result


def compare_importance_methods(tree_importance_df, perm_importance_df, top_n=10):
    """
    Compare tree-based and permutation-based importance rankings.
    
    Parameters
    ----------
    tree_importance_df : pd.DataFrame
        Tree-based importance (from get_feature_importance_df)
    perm_importance_df : pd.DataFrame
        Permutation importance (from compute_permutation_importance)
    top_n : int, default=10
        Number of top features to compare
        
    Returns
    -------
    pd.DataFrame
        Comparison table showing both importance measures
    """
    # Get top N from each method
    tree_top = tree_importance_df.head(top_n).copy()
    perm_top = perm_importance_df.head(top_n).copy()
    
    # Merge on feature name
    comparison = pd.merge(
        tree_top[['Feature', 'Importance']],
        perm_top[['Feature', 'Importance', 'Std']],
        on='Feature',
        how='outer',
        suffixes=('_Tree', '_Perm')
    )
    
    # Fill NaN for features that don't appear in both top-N lists
    comparison = comparison.fillna(0)
    
    # Sort by average of both importances
    comparison['Avg_Importance'] = (comparison['Importance_Tree'] + comparison['Importance_Perm']) / 2
    comparison = comparison.sort_values('Avg_Importance', ascending=False)
    
    return comparison[['Feature', 'Importance_Tree', 'Importance_Perm', 'Std']]


def get_shap_values(model, X):
    """
    Compute SHAP values for model interpretation.
    
    Requires: pip install shap
    
    Parameters
    ----------
    model : sklearn estimator
        Trained tree-based model (RandomForest, GradientBoosting, etc.)
    X : pd.DataFrame or np.ndarray
        Feature matrix
        
    Returns
    -------
    tuple
        (explainer, shap_values)
        - explainer: SHAP TreeExplainer object
        - shap_values: SHAP values (list of arrays for multiclass)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return explainer, shap_values


def get_shap_importance(shap_values, feature_names, class_idx=None):
    """
    Get feature importance from SHAP values (mean absolute SHAP value).
    
    Parameters
    ----------
    shap_values : np.ndarray or list of np.ndarray
        SHAP values from explainer
    feature_names : list
        Names of features
    class_idx : int, optional
        For multiclass, which class to compute importance for.
        If None and multiclass, returns dict with importance per class.
        
    Returns
    -------
    pd.DataFrame or dict of pd.DataFrame
        Feature importance based on SHAP values
    """
    # Handle multiclass (list of arrays)
    if isinstance(shap_values, list):
        if class_idx is not None:
            # Single class
            importance = np.abs(shap_values[class_idx]).mean(axis=0)
            return pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Importance': importance
            }).sort_values('SHAP_Importance', ascending=False)
        else:
            # All classes
            result = {}
            for idx in range(len(shap_values)):
                importance = np.abs(shap_values[idx]).mean(axis=0)
                result[f'Class_{idx}'] = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Importance': importance
                }).sort_values('SHAP_Importance', ascending=False)
            return result
    else:
        # Binary or single output
        importance = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': importance
        }).sort_values('SHAP_Importance', ascending=False)