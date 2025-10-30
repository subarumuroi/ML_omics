"""
Model evaluation utilities and metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)


def get_classification_report(model, X, y, target_names=None, output_dict=False):
    """
    Generate classification report from a trained model.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model with predict method
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        True labels
    target_names : list, optional
        Class names for report
    output_dict : bool, default=False
        If True, return as dict instead of string
        
    Returns
    -------
    str or dict
        Classification report
    """
    y_pred = model.predict(X)
    return classification_report(
        y, y_pred, 
        target_names=target_names,
        output_dict=output_dict
    )


def get_confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list, optional
        List of label values to include
        
    Returns
    -------
    np.ndarray
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_cv_metrics(y_true, y_pred):
    """
    Compute accuracy and balanced accuracy metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns
    -------
    dict
        Dictionary with accuracy and balanced_accuracy
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }


def create_results_summary(cv_results, perm_test_results=None, feature_info=None):
    """
    Create comprehensive summary dictionary of model results.
    
    Parameters
    ----------
    cv_results : dict
        Cross-validation results from RFClassifierCV
    perm_test_results : dict, optional
        Permutation test results
    feature_info : dict, optional
        Information about features (e.g., number of features, names)
        
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
            'permutation_observed': perm_test_results['observed_accuracy'],
        })
    
    if feature_info is not None:
        summary.update(feature_info)
    
    return summary


def compare_models(results_list, model_names=None):
    """
    Compare multiple model results in a table.
    
    Parameters
    ----------
    results_list : list of dict
        List of result dictionaries (from create_results_summary)
    model_names : list of str, optional
        Names for each model
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(results_list))]
    
    comparison_data = {
        'Model': model_names,
        'CV Accuracy': [r['cv_accuracy'] for r in results_list],
        'CV Accuracy Std': [r['cv_accuracy_std'] for r in results_list],
        'CV Balanced Acc': [r['cv_balanced_accuracy'] for r in results_list],
        'CV Balanced Std': [r['cv_balanced_accuracy_std'] for r in results_list],
    }
    
    # Add permutation test results if available
    if all('permutation_p_value' in r for r in results_list):
        comparison_data['Perm p-value'] = [r['permutation_p_value'] for r in results_list]
    
    return pd.DataFrame(comparison_data)