"""
Random Forest classifier with cross-validation support.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


class RFClassifierCV:
    """
    Random Forest Classifier with built-in Cross-Validation.
    
    This class wraps sklearn's RandomForestClassifier and adds convenient
    cross-validation functionality with result tracking.
    """
    
    def __init__(self, n_estimators=100, max_depth=3, random_state=42, **kwargs):
        """
        Initialize RF classifier.
        
        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, default=3
            Maximum depth of trees
        random_state : int, default=42
            Random state for reproducibility
        **kwargs : dict
            Additional parameters passed to RandomForestClassifier
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self.cv_results = None
        self.confusion_matrix = None
        self._is_fitted = False
        
    def cross_validate(self, X, y, n_splits=None, random_state=42, verbose=True):
        """
        Perform stratified k-fold cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels (encoded as integers)
        n_splits : int, optional
            Number of CV folds. If None, uses minimum class count
        random_state : int, default=42
            Random state for reproducible shuffling
        verbose : bool, default=True
            Print fold results
            
        Returns
        -------
        dict
            CV metrics and predictions
        """
        n_splits = n_splits or np.bincount(y).min()
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        scores, bal_scores, all_true, all_pred = [], [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            
            self.model.fit(X_train, y[train_idx])
            y_pred = self.model.predict(X_test)
            y_true = y[test_idx]
            
            all_true.extend(y_true)
            all_pred.extend(y_pred)
            
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            scores.append(acc)
            bal_scores.append(bal_acc)
            
            if verbose:
                print(f"Fold {fold+1}: Acc={acc:.3f}, Bal={bal_acc:.3f}")
        
        self.cv_results = {
            'scores': scores,
            'balanced_scores': bal_scores,
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'mean_balanced_accuracy': np.mean(bal_scores),
            'std_balanced_accuracy': np.std(bal_scores),
            'all_true': np.array(all_true),
            'all_pred': np.array(all_pred),
        }
        
        self.confusion_matrix = confusion_matrix(all_true, all_pred)
        
        if verbose:
            print(f"\nCV: {self.cv_results['mean_accuracy']:.3f}±{self.cv_results['std_accuracy']:.3f}")
            print(f"Balanced: {self.cv_results['mean_balanced_accuracy']:.3f}±{self.cv_results['std_balanced_accuracy']:.3f}")
        
        return self.cv_results
    
    def fit(self, X, y):
        """
        Fit model on full dataset.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
            
        Returns
        -------
        self
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted probabilities for each class
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importances(self, feature_names=None):
        """
        Get feature importances from trained model.
        
        Parameters
        ----------
        feature_names : list, optional
            Names of features
            
        Returns
        -------
        dict or np.ndarray
            Feature importances (dict if names provided, else array)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing feature importances.")
            
        importances = self.model.feature_importances_
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return importances


def train_evaluate_model(X, y, n_estimators=100, max_depth=3, n_splits=None, 
                         random_state=42, verbose=True, **kwargs):
    """
    Convenience function to train and evaluate RF model with CV.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    n_estimators : int, default=100
        Number of trees
    max_depth : int, default=3
        Maximum tree depth
    n_splits : int, optional
        Number of CV folds (auto-detected if None)
    random_state : int, default=42
        Random state
    verbose : bool, default=True
        Print results
    **kwargs : dict
        Additional parameters for RandomForestClassifier
        
    Returns
    -------
    RFClassifierCV
        Trained classifier with CV results
    """
    clf = RFClassifierCV(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
    
    # Run cross-validation
    clf.cross_validate(X, y, n_splits=n_splits, verbose=verbose)
    
    # Fit on full dataset
    clf.fit(X, y)
    
    return clf