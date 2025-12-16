"""
Partial Least Squares Discriminant Analysis (PLS-DA) model for
metabolomics data, leveraging PLSRegression and component analysis.
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression

# Note: You might need to adjust the path below if your structure differs
# For example: from ..models.evaluation import compute_cv_metrics
# For standalone test:
# from models.evaluation import compute_cv_metrics # Assuming a standard import path

def train_evaluate_plsda(X, y_raw, n_components=None, scaler='standard', log_transform=True,
                         n_splits=5, random_state=42, verbose=True):
    """
    Train and evaluate PLS-DA model with cross-validation.
    
    PLS-DA uses PLSRegression combined with a Logistic Regression classifier on
    the target classes, which is suitable for metabolomics data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y_raw : pd.Series or np.ndarray
        Raw, non-encoded categorical target (e.g., 'Green', 'Ripe').
    n_components : int, optional
        Number of PLS components. Automatically set if None.
    scaler : str, default='standard'
        Scaling method.
    log_transform : bool, default=True
        Apply log transformation.
    n_splits : int, default=5
        Number of CV folds.
    random_state : int, default=42
        Random state.
    verbose : bool, default=True
        Print results.
        
    Returns
    -------
    dict
        Dictionary with trained model and CV results.
    """
    if n_components is None:
        # Heuristic for components
        n_components = min(X.shape[1], 10)
        
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    feature_names = X.columns.tolist()

    # Encode raw Y for stratification
    y_le = LabelEncoder().fit_transform(y_raw)
    
    # Store results
    cv_scores = []
    cv_mae = []
    all_true = []
    all_pred = []
    
    scaler_cls = MinMaxScaler if scaler == 'minmax' else StandardScaler
        
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_le)):
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_le[train_idx], y_le[test_idx]

        # Preprocessing (within CV loop for proper scaling)
        s = scaler_cls()
        
        # Apply log transformation (handling potential zeros)
        X_train_proc = np.log10(X_train + 1e-10) if log_transform else X_train
        X_test_proc = np.log10(X_test + 1e-10) if log_transform else X_test
        
        X_train_scaled = s.fit_transform(X_train_proc)
        X_test_scaled = s.transform(X_test_proc)

        # 1. PLS Regression (on OHE target for optimal component calculation)
        ohe = OneHotEncoder(sparse_output=False, categories='auto', drop='first')
        y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
        
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train_scaled, y_train_ohe)
        
        # 2. Transform and Classify (using Logistic Regression)
        X_train_pls = pls.transform(X_train_scaled)
        X_test_pls = pls.transform(X_test_scaled)

        lr_clf = LogisticRegression(solver='liblinear', random_state=random_state)
        lr_clf.fit(X_train_pls, y_train)
        y_pred = lr_clf.predict(X_test_pls)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_scores.append(acc)
        cv_mae.append(mae)
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        
        if verbose:
            print(f"Fold {fold+1}: Accuracy={acc:.3f}, MAE={mae:.3f}")

    # Train final model on all data for feature interpretation
    s = scaler_cls()
    X_proc_all = np.log10(X + 1e-10) if log_transform else X
    X_scaled_all = s.fit_transform(X_proc_all)
    y_ohe_all = OneHotEncoder(sparse_output=False, categories='auto', drop='first').fit_transform(y_le.reshape(-1, 1))
    
    final_pls = PLSRegression(n_components=n_components)
    final_pls.fit(X_scaled_all, y_ohe_all)
    
    # Feature Interpretation (mean absolute loading)
    pls_loadings = pd.DataFrame(final_pls.x_weights_, index=feature_names, 
                                columns=[f'Component_{i+1}' for i in range(n_components)])
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': pls_loadings.abs().mean(axis=1).values
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\nMean CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"Mean CV MAE: {np.mean(cv_mae):.3f} ± {np.std(cv_mae):.3f}")

    results = {
        'model': final_pls,
        'feature_importance': feature_importance_df,
        'cv_scores': cv_scores,
        'cv_mae': cv_mae,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'mean_mae': np.mean(cv_mae),
        'std_mae': np.std(cv_mae),
        'all_true': np.array(all_true),
        'all_pred': np.array(all_pred),
        'n_components': n_components
    }
    
    return results