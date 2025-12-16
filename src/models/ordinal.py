"""
Ordinal regression models for ordered categorical outcomes.

Handles cases where the target has a natural ordering (e.g., Green < Ripe < Overripe).
Requires: pip install mord
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, mean_absolute_error

try:
    from mord import LogisticAT, LogisticIT, LogisticSE
    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False


class OrdinalClassifier:
    """
    Ordinal regression classifier with preprocessing.
    
    Automatically handles log transformation and scaling of features
    for ordinal regression tasks.
    """
    
    def __init__(self, model_type='LogisticAT', scaler='minmax', log_transform=True):
        """
        Initialize ordinal classifier.
        
        Parameters
        ----------
        model_type : str, default='LogisticAT'
            Type of ordinal model: 'LogisticAT', 'LogisticIT', or 'LogisticSE'
            - LogisticAT: All-Threshold variant (most common)
            - LogisticIT: Immediate-Threshold variant
            - LogisticSE: Squared-Error variant
        scaler : str, default='minmax'
            Scaling method: 'minmax' or 'standard'
        log_transform : bool, default=True
            Apply log10 transformation before scaling
        """
        if not MORD_AVAILABLE:
            raise ImportError("mord package not installed. Install with: pip install mord")
        
        self.model_type = model_type
        self.scaler_type = scaler
        self.log_transform = log_transform
        
        # Initialize model
        if model_type == 'LogisticAT':
            self.model = LogisticAT()
        elif model_type == 'LogisticIT':
            self.model = LogisticIT()
        elif model_type == 'LogisticSE':
            self.model = LogisticSE()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Initialize scaler
        if scaler == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler}")
        
        self._is_fitted = False
        self.feature_names_ = None
        
    def _preprocess(self, X, fit=False):
        """Apply log transformation and scaling."""
        X_processed = X.copy() if isinstance(X, pd.DataFrame) else X
        
        if self.log_transform:
            X_processed = np.log10(X_processed + 1e-10)  # Handle zeros
        
        X_processed = self.scaler.fit_transform(X_processed) if fit else self.scaler.transform(X_processed)
        
        return X_processed
    
    def fit(self, X, y):
        """
        Fit ordinal regression model.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : np.ndarray
            Ordinal target (encoded as 0, 1, 2, ...)
            
        Returns
        -------
        self
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        
        # Preprocess
        X_processed = self._preprocess(X, fit=True)
        
        # Fit model
        self.model.fit(X_processed, y)
        self._is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Predict ordinal class.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted ordinal classes
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Preprocess and predict
        X_processed = self._preprocess(X, fit=False)
        return self.model.predict(X_processed)
    
    def get_coefficients(self, feature_names=None):
        """
        Get feature coefficients from trained model.
        
        Parameters
        ----------
        feature_names : list, optional
            Names of features (uses stored names if available)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with Feature, Coefficient, and Abs_Coefficient columns
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        coefficients = self.model.coef_
        
        # Use stored or provided feature names
        if feature_names is None:
            feature_names = self.feature_names_
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(coefficients))]
        
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values(by='Abs_Coefficient', ascending=False)
        
        return coef_df
    
    def score(self, X, y):
        """
        Compute accuracy score.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
            
        Returns
        -------
        float
            Accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


def train_evaluate_ordinal(X, y, model_type='LogisticAT', scaler='minmax', 
                           log_transform=True, n_splits=None, random_state=42,
                           verbose=True):
    """
    Train and evaluate ordinal regression model with cross-validation.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        Ordinal target (0, 1, 2, ...)
    model_type : str, default='LogisticAT'
        Type of ordinal model
    scaler : str, default='minmax'
        Scaling method
    log_transform : bool, default=True
        Apply log transformation
    n_splits : int, optional
        Number of CV folds (auto-detected if None)
    random_state : int, default=42
        Random state
    verbose : bool, default=True
        Print results
        
    Returns
    -------
    dict
        Dictionary with trained model and CV results
    """
    # Determine CV splits
    N_min_class = np.bincount(y).min()
    N_total = X.shape[0]
    
    # FIX: Handle n_splits=None and cap it by N_min_class
    if n_splits is None:
        k_actual = N_min_class # Use max possible splits (min class size)
    else:
        k_actual = n_splits
        
    if k_actual == N_total:
        cv = LeaveOneOut()
    elif k_actual > N_min_class:
        # If requested splits (or None default) > min class size, cap it.
        k_actual = N_min_class
        cv = StratifiedKFold(n_splits=k_actual, shuffle=True, random_state=random_state)
        if verbose:
             print(f"Warning: Requested n_splits ({n_splits}) > Min class size ({N_min_class}). Capping at {k_actual}.")
    else:
        cv = StratifiedKFold(n_splits=k_actual, shuffle=True, random_state=random_state)
    
    # Store feature names
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    
    # Cross-validation
    cv_scores = []
    cv_mae = []
    all_true = []
    all_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Split data
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        clf = OrdinalClassifier(model_type=model_type, scaler=scaler, log_transform=log_transform)
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_scores.append(acc)
        cv_mae.append(mae)
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        
        if verbose:
            print(f"Fold {fold+1}: Accuracy={acc:.3f}, MAE={mae:.3f}")
    
    if verbose:
        print(f"\nMean CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"Mean CV MAE: {np.mean(cv_mae):.3f} ± {np.std(cv_mae):.3f}")
    
    # Train final model on all data
    final_model = OrdinalClassifier(model_type=model_type, scaler=scaler, log_transform=log_transform)
    final_model.fit(X, y)
    
    # Get coefficients
    coef_df = final_model.get_coefficients(feature_names)
    
    results = {
        'model': final_model,
        'coefficients': coef_df,
        'cv_scores': cv_scores,
        'cv_mae': cv_mae,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'mean_mae': np.mean(cv_mae),
        'std_mae': np.std(cv_mae),
        'all_true': np.array(all_true),
        'all_pred': np.array(all_pred),
    }
    
    return results


def compare_ordinal_models(X, y, model_types=['LogisticAT', 'LogisticIT', 'LogisticSE'],
                           n_splits=None, verbose=True):
    """
    Compare different ordinal regression models.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        Ordinal target
    model_types : list, default=['LogisticAT', 'LogisticIT', 'LogisticSE']
        Models to compare
    n_splits : int, optional
        Number of CV folds
    verbose : bool, default=True
        Print results
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    results_list = []
    
    for model_type in model_types:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {model_type}")
            print(f"{'='*60}")
        
        results = train_evaluate_ordinal(
            X, y, 
            model_type=model_type,
            n_splits=n_splits,
            verbose=verbose
        )
        
        results_list.append({
            'Model': model_type,
            'Accuracy': results['mean_accuracy'],
            'Accuracy_Std': results['std_accuracy'],
            'MAE': results['mean_mae'],
            'MAE_Std': results['std_mae'],
        })
    
    comparison = pd.DataFrame(results_list)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}")
        print(comparison)
    
    return comparison


def encode_ordinal_target(target, categories=None, ordinal_map=None):
    """
    Encode target as ordinal values.
    
    Parameters
    ----------
    target : pd.Series
        Target variable
    categories : list, optional
        Ordered list of categories (e.g., ['Green', 'Ripe', 'Overripe'])
    ordinal_map : dict, optional
        Manual mapping (e.g., {'Green': 0, 'Ripe': 1, 'Overripe': 2})
        
    Returns
    -------
    np.ndarray
        Encoded ordinal values
    """
    if ordinal_map is not None:
        return target.map(ordinal_map).values
    elif categories is not None:
        ordinal_map = {cat: i for i, cat in enumerate(categories)}
        return target.map(ordinal_map).values
    else:
        raise ValueError("Must provide either categories or ordinal_map")