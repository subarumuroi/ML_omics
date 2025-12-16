"""
Support Vector Machine (SVM) model for classification.
Includes hyperparameter tuning via GridSearchCV.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def train_evaluate_svm(X, y, kernel='rbf', scaler='minmax', log_transform=True,
                       n_splits=5, random_state=42, verbose=True):
    """
    Train and evaluate SVM model with cross-validation and hyperparameter tuning.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Ordinal target (encoded as 0, 1, 2, ...).
    kernel : str, default='rbf'
        Kernel type: 'linear' or 'rbf'.
    scaler : str, default='minmax'
        Scaling method.
    log_transform : bool, default=True
        Apply log transformation.
    n_splits : int, default=5
        Number of CV folds. (This controls the OUTER loop)
    random_state : int, default=42
        Random state.
    verbose : bool, default=True
        Print results.
        
    Returns
    -------
    dict
        Dictionary with trained model and CV results.
    """
    # n_splits will be 3, as passed by general_badata_workflow.py for small data
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    feature_names = X.columns.tolist()

    # 1. Preprocessing Setup
    scaler_cls = MinMaxScaler if scaler == 'minmax' else StandardScaler
    s = scaler_cls()
    X_proc_all = np.log10(X + 1e-10) if log_transform else X
    
    # Scale features
    X_scaled_all = s.fit_transform(X_proc_all)
    
    # 2. Hyperparameter Grid
    if kernel == 'linear':
        param_grid = {'C': [0.1, 1, 10]}
    elif kernel == 'rbf':
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]}
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'linear' or 'rbf'.")
        
    # 3. Cross-validation (Outer loop)
    cv_scores = []
    cv_mae = []
    all_true = []
    all_pred = []

    # Inner CV folds must be 2 because the minimum class size in the training fold (N=6, 2:2:2) is 2.
    INNER_CV_SPLITS = 2

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled_all, y)):
        X_train, X_test = X_scaled_all[train_idx], X_scaled_all[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Use GridSearchCV for optimal parameter selection within the training fold (Inner loop)
        clf = SVC(kernel=kernel, random_state=random_state)
        
        # INNER CV FIX: Explicitly set cv=2 for stability
        grid_search = GridSearchCV(clf, param_grid, cv=INNER_CV_SPLITS, scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        best_clf = grid_search.best_estimator_
        
        # Predict on test set
        y_pred = best_clf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_scores.append(acc)
        cv_mae.append(mae)
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        
        if verbose:
            print(f"Fold {fold+1}: Accuracy={acc:.3f}, MAE={mae:.3f} (Best Params: {grid_search.best_params_})")

    # 4. Train final model on all data
    final_clf = SVC(kernel=kernel, random_state=random_state)
    
    # FINAL GRID FIX: Set cv=n_splits (which is 3) as the full dataset has min class size 3
    final_grid = GridSearchCV(final_clf, param_grid, cv=n_splits, scoring='accuracy', n_jobs=-1, verbose=0)
    final_grid.fit(X_scaled_all, y)
    
    final_model = final_grid.best_estimator_
    
    # 5. Feature Importance (Linear Kernel only)
    feature_importance_df = pd.DataFrame()
    if kernel == 'linear' and hasattr(final_model, 'coef_'):
        # Mean absolute coefficient across the one-vs-rest classifiers
        coefs = np.abs(final_model.coef_).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefs
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\nMean CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"Mean CV MAE: {np.mean(cv_mae):.3f} ± {np.std(cv_mae):.3f}")

    results = {
        'model': final_model,
        'feature_importance': feature_importance_df, 
        'cv_scores': cv_scores,
        'cv_mae': cv_mae,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'mean_mae': np.mean(cv_mae),
        'std_mae': np.std(cv_mae),
        'all_true': np.array(all_true),
        'all_pred': np.array(all_pred),
        'best_params': final_grid.best_params_
    }
    
    return results