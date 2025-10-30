"""
ML model training and evaluation module.
Handles Random Forest classification with cross-validation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


class RFClassifierCV:
    """Random Forest Classifier with Cross-Validation support."""
    
    def __init__(self, n_estimators=100, max_depth=3, random_state=42):
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
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.cv_results = None
        self.confusion_matrix = None
        
    def cross_validate(self, X, y, n_splits=None, verbose=True):
        """
        Perform stratified k-fold cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        n_splits : int, optional
            Number of CV folds. If None, uses min class count
        verbose : bool, default=True
            Print fold results
            
        Returns
        -------
        dict
            Dictionary containing CV metrics and predictions
        """
        if n_splits is None:
            n_splits = np.bincount(y).min()
            
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
        
        scores = []
        bal_scores = []
        all_true = []
        all_pred = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Handle both DataFrame and array inputs
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
                print(f"Fold {fold+1} accuracy: {acc:.3f} | balanced accuracy: {bal_acc:.3f}")
                print(f"  True: {y_true} | Pred: {y_pred}")
        
        # Store results
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
        
        # Calculate confusion matrix
        self.confusion_matrix = confusion_matrix(all_true, all_pred)
        
        if verbose:
            print(f"\nMean CV accuracy: {self.cv_results['mean_accuracy']:.3f} ± {self.cv_results['std_accuracy']:.3f}")
            print(f"Mean balanced accuracy: {self.cv_results['mean_balanced_accuracy']:.3f} ± {self.cv_results['std_balanced_accuracy']:.3f}")
        
        return self.cv_results
    
    def fit(self, X, y):
        """Fit model on full dataset."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
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
        importances = self.model.feature_importances_
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return importances
    
    def get_classification_report(self, X, y, target_names=None):
        """
        Generate classification report.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
        target_names : list, optional
            Class names for report
            
        Returns
        -------
        str
            Classification report
        """
        y_pred = self.predict(X)
        return classification_report(y, y_pred, target_names=target_names)


def train_evaluate_model(X, y, n_estimators=100, max_depth=3, n_splits=None, 
                         random_state=42, verbose=True):
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
        Number of CV folds
    random_state : int, default=42
        Random state
    verbose : bool, default=True
        Print results
        
    Returns
    -------
    RFClassifierCV
        Trained classifier with CV results
    """
    clf = RFClassifierCV(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    clf.cross_validate(X, y, n_splits=n_splits, verbose=verbose)
    clf.fit(X, y)  # Fit on full dataset
    
    return clf