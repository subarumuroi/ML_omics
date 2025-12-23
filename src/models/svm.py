"""
Support Vector Machine (SVM) model for classification.
Includes hyperparameter tuning via GridSearchCV.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, accuracy_score
# LeaveOneOut is already imported, perfect!
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def train_evaluate_svm(X, y, kernel='rbf', scaler='minmax', log_transform=True,
                       n_splits=5, random_state=42, verbose=True):
    """
    Train and evaluate SVM model with cross-validation and hyperparameter tuning.
    
    ... (docstring remains the same)
    """
    # Determine CV splits
    N_min_class = np.bincount(y).min()
    N_total = X.shape[0]
    if n_splits == N_total:
        cv = LeaveOneOut()
        final_grid_cv = LeaveOneOut()
    elif n_splits > N_min_class:
        cv = StratifiedKFold(n_splits=N_min_class, shuffle=True, random_state=random_state)
        final_grid_cv = cv
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        final_grid_cv = cv
    
    feature_names = X.columns.tolist()

    # 1. Preprocessing Setup (remains the same)
    scaler_cls = MinMaxScaler if scaler == 'minmax' else StandardScaler
    s = scaler_cls()
    X_proc_all = np.log10(X + 1e-10) if log_transform else X
    
    # Scale features
    X_scaled_all = s.fit_transform(X_proc_all)
    
    # 2. Hyperparameter Grid (remains the same)
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
    # Note: We keep this fixed at 2 for maximum stability in your small dataset.
    INNER_CV_SPLITS = 2 

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled_all, y)): # 'cv' is now LeaveOneOut if n_splits=9
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
        
        # Metrics (remains the same)
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
    
    # FINAL GRID FIX: Use the conditionally defined final_grid_cv (which is LeaveOneOut if n_splits=9)
    final_grid = GridSearchCV(final_clf, param_grid, cv=final_grid_cv, scoring='accuracy', n_jobs=-1, verbose=0)
    final_grid.fit(X_scaled_all, y)
    
    final_model = final_grid.best_estimator_
    
    # 5. Feature Importance (Linear Kernel only) (remains the same)
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