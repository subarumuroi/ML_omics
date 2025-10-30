"""
Statistical tests module for model validation.
Includes permutation tests for assessing classification significance.
"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def permutation_test(model, X, y, cv, n_permutations=1000, random_state=None, verbose=True):
    """
    Perform permutation test to assess model significance.
    
    Tests the null hypothesis that there's no relationship between features
    and labels by randomly shuffling labels and comparing performance.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate (must have fit/predict methods)
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    cv : sklearn CV splitter
        Cross-validation strategy
    n_permutations : int, default=1000
        Number of permutations to perform
    random_state : int, optional
        Base random state for reproducibility
    verbose : bool, default=True
        Print progress updates
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'perm_accuracies': array of permutation test accuracies
        - 'p_value': permutation test p-value
        - 'observed_accuracy': observed mean CV accuracy
        - 'perm_mean': mean of permutation accuracies
        - 'perm_std': std of permutation accuracies
    """
    # First, get observed accuracy with true labels
    observed_scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
        
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_test)
        acc = accuracy_score(y[test_idx], y_pred)
        observed_scores.append(acc)
    
    observed_acc = np.mean(observed_scores)
    
    # Permutation test
    perm_accuracies = []
    
    for i in range(n_permutations):
        if verbose and (i + 1) % 100 == 0:
            print(f"Permutation {i+1}/{n_permutations}")
        
        # Shuffle labels
        seed = random_state + i if random_state is not None else None
        y_perm = shuffle(y, random_state=seed)
        
        # Run CV with permuted labels
        perm_scores = []
        for train_idx, test_idx in cv.split(X, y_perm):
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            
            model.fit(X_train, y_perm[train_idx])
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_perm[test_idx], y_pred)
            perm_scores.append(acc)
        
        perm_accuracies.append(np.mean(perm_scores))
    
    perm_accuracies = np.array(perm_accuracies)
    
    # Calculate p-value
    p_value = (np.sum(perm_accuracies >= observed_acc) + 1) / (n_permutations + 1)
    
    results = {
        'perm_accuracies': perm_accuracies,
        'p_value': p_value,
        'observed_accuracy': observed_acc,
        'perm_mean': np.mean(perm_accuracies),
        'perm_std': np.std(perm_accuracies),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Permutation Test Results")
        print(f"{'='*60}")
        print(f"Observed accuracy: {observed_acc:.3f}")
        print(f"Mean permuted accuracy: {results['perm_mean']:.3f} ± {results['perm_std']:.3f}")
        print(f"p-value: {p_value:.4f}")
        print(f"{'='*60}\n")
    
    return results


def permutation_test_with_feature_subset(model, X_full, feature_names, y, 
                                        n_splits=None, n_permutations=1000, 
                                        random_state=42, verbose=True):
    """
    Convenience function to run permutation test on a feature subset.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate
    X_full : pd.DataFrame
        Full feature matrix
    feature_names : list
        Names of features to use
    y : np.ndarray
        Target labels
    n_splits : int, optional
        Number of CV folds
    n_permutations : int, default=1000
        Number of permutations
    random_state : int, default=42
        Random state
    verbose : bool, default=True
        Print results
        
    Returns
    -------
    dict
        Permutation test results
    """
    # Extract feature subset
    X_subset = X_full[feature_names]
    
    # Setup CV
    if n_splits is None:
        n_splits = np.bincount(y).min()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    
    if verbose:
        print(f"Running permutation test with {len(feature_names)} features")
        print(f"Features: {feature_names}")
    
    return permutation_test(
        model=model,
        X=X_subset,
        y=y,
        cv=cv,
        n_permutations=n_permutations,
        random_state=random_state,
        verbose=verbose
    )


def compare_permutation_tests(results_list, labels=None):
    """
    Compare multiple permutation test results.
    
    Parameters
    ----------
    results_list : list of dict
        List of permutation test results
    labels : list of str, optional
        Labels for each test
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    import pandas as pd
    
    if labels is None:
        labels = [f"Test {i+1}" for i in range(len(results_list))]
    
    comparison = pd.DataFrame({
        'Test': labels,
        'Observed Accuracy': [r['observed_accuracy'] for r in results_list],
        'Perm Mean': [r['perm_mean'] for r in results_list],
        'Perm Std': [r['perm_std'] for r in results_list],
        'p-value': [r['p_value'] for r in results_list],
    })
    
    return comparison