"""
Partial Least Squares Discriminant Analysis (PLS-DA) model for
metabolomics data, leveraging PLSRegression and component analysis.
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression

# Note: You might need to adjust the path below if your structure differs
# from models.evaluation import compute_cv_metrics # If needed

def train_evaluate_plsda(X, y_raw, n_components=None, scaler='standard', log_transform=True,
                         n_splits=5, random_state=42, verbose=True):
    """
    Train and evaluate PLS-DA model with cross-validation.
    """
    N_total = X.shape[0]
    feature_names = X.columns.tolist()
    
    # 1. Target Encoding (MUST be done first to allow class counting)
    le = LabelEncoder()
    y_le = le.fit_transform(y_raw)
    
    # 2. Determine Robust CV Splits (k_actual)
    N_min_class = np.bincount(y_le).min()
    
    if n_splits == N_total:
        # User explicitly requested LOOCV
        cv = LeaveOneOut()
        k_actual = N_total
        N_test_size = 1
    elif n_splits > N_min_class:
        # Requested splits too high for stratification (e.g., 9 splits on min class size 4)
        k_actual = N_min_class
        cv = StratifiedKFold(n_splits=k_actual, shuffle=True, random_state=random_state)
        N_test_size = N_total // k_actual
        if verbose:
            print(f"Warning: Requested n_splits ({n_splits}) > Min class size ({N_min_class}). Using Stratified {k_actual}-Fold CV.")
    else:
        # Use the requested safe n_splits (e.g., 5 splits on min class size 6)
        k_actual = n_splits
        cv = StratifiedKFold(n_splits=k_actual, shuffle=True, random_state=random_state)
        N_test_size = N_total // k_actual
        
    # 3. Determine Robust n_components Cap (N_min_train)
    
    # Calculate the smallest training set size based on the ACTUAL split scheme
    N_min_train = N_total - N_test_size
    
    if n_components is None:
        # Initial heuristic: min of features and a max of 10
        n_components = min(X.shape[1], 10)
        
    # Cap n_components by the minimum possible training set size
    if n_components > N_min_train:
        if verbose:
            print(f"Warning: n_components ({n_components}) limited by minimum training set size ({N_min_train} for {k_actual}-Fold CV). Capping at {N_min_train}.")
        n_components = N_min_train
        
    # Store results
    cv_scores = []
    cv_mae = []
    all_true = []
    all_pred = []
    
    scaler_cls = MinMaxScaler if scaler == 'minmax' else StandardScaler
        
    # --- CROSS-VALIDATION LOOP ---
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_le)):
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        # Use integer-encoded y_le
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
        
        # n_components is now guaranteed to be safe
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train_scaled, y_train_ohe)
        
        # 2. Transform and Classify (using Logistic Regression)
        X_train_pls = pls.transform(X_train_scaled)
        X_test_pls = pls.transform(X_test_scaled)

        # Note: Scikit-learn issues a warning about 'liblinear' multiclass solver, 
        # but it functions as intended for now.
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

    # --- FINAL MODEL TRAINING & INTERPRETATION ---
    # Train final model on all data for feature interpretation
    s = scaler_cls()
    X_proc_all = np.log10(X + 1e-10) if log_transform else X
    X_scaled_all = s.fit_transform(X_proc_all)
    y_ohe_all = OneHotEncoder(sparse_output=False, categories='auto', drop='first').fit_transform(y_le.reshape(-1, 1))
    
    # n_components is now guaranteed to be safe
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