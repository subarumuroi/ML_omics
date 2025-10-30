"""
Visualization functions specific to ordinal regression models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ordinal_coefficients(coef_df, top_n=10, figsize=(10, 8), 
                              palette='coolwarm', save_path=None):
    """
    Plot coefficients from ordinal regression model.
    
    Parameters
    ----------
    coef_df : pd.DataFrame
        DataFrame with Feature and Coefficient columns (from OrdinalClassifier.get_coefficients())
    top_n : int, default=10
        Number of top features to plot
    figsize : tuple, default=(10, 8)
        Figure size
    palette : str, default='coolwarm'
        Color palette
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes
    """
    data = coef_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color mapping based on sign of coefficient
    colors = ['#d7191c' if c < 0 else '#2b83ba' for c in data['Coefficient']]
    
    sns.barplot(
        data=data,
        x='Coefficient',
        y='Feature',
        palette=colors,
        ax=ax
    )
    
    ax.set_title(f'Top {top_n} Features Predicting Ripeness (Ordinal Regression)')
    ax.set_xlabel('Coefficient (← decreases ripeness | increases ripeness →)')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_ordinal_confusion_matrix(y_true, y_pred, class_labels=None, 
                                  figsize=(8, 6), cmap='Blues', save_path=None):
    """
    Plot confusion matrix for ordinal predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True ordinal labels
    y_pred : np.ndarray
        Predicted ordinal labels
    class_labels : list, optional
        Class names (e.g., ['Green', 'Ripe', 'Overripe'])
    figsize : tuple, default=(8, 6)
        Figure size
    cmap : str, default='Blues'
        Colormap
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_labels is None:
        class_labels = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=cmap, values_format='d', ax=ax)
    ax.set_title('Ordinal Regression Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_ordinal_predictions_vs_true(y_true, y_pred, class_labels=None,
                                     figsize=(8, 6), save_path=None):
    """
    Plot predicted vs true ordinal values.
    
    Shows how predictions align with true values, useful for assessing
    if model makes systematic errors (e.g., always predicting one class too high).
    
    Parameters
    ----------
    y_true : np.ndarray
        True ordinal labels
    y_pred : np.ndarray
        Predicted ordinal labels
    class_labels : list, optional
        Class names
    figsize : tuple, default=(8, 6)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add jitter for visualization
    jitter = 0.1
    y_true_jitter = y_true + np.random.normal(0, jitter, size=len(y_true))
    y_pred_jitter = y_pred + np.random.normal(0, jitter, size=len(y_pred))
    
    # Scatter plot
    ax.scatter(y_true_jitter, y_pred_jitter, alpha=0.6, s=100)
    
    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction', linewidth=2)
    
    # Labels
    ax.set_xlabel('True Class')
    ax.set_ylabel('Predicted Class')
    ax.set_title('Predicted vs True Ordinal Values')
    
    # Set integer ticks
    unique_vals = np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(unique_vals)
    ax.set_yticks(unique_vals)
    
    # Use class labels if provided
    if class_labels is not None:
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
    
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_ordinal_cv_performance(cv_scores, cv_mae, figsize=(12, 5), save_path=None):
    """
    Plot cross-validation performance for ordinal model.
    
    Parameters
    ----------
    cv_scores : list
        Accuracy scores per fold
    cv_mae : list
        MAE scores per fold
    figsize : tuple, default=(12, 5)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    folds = np.arange(1, len(cv_scores) + 1)
    
    # Accuracy plot
    axes[0].plot(folds, cv_scores, marker='o', linewidth=2, markersize=8)
    axes[0].axhline(np.mean(cv_scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(cv_scores):.3f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Cross-Validation Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(folds)
    
    # MAE plot
    axes[1].plot(folds, cv_mae, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].axhline(np.mean(cv_mae), color='red', linestyle='--',
                    label=f'Mean: {np.mean(cv_mae):.3f}')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Cross-Validation MAE')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(folds)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_ordinal_model_comparison(comparison_df, figsize=(10, 6), save_path=None):
    """
    Plot comparison of different ordinal models.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison table from compare_ordinal_models()
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    models = comparison_df['Model']
    
    # Accuracy comparison
    axes[0].bar(models, comparison_df['Accuracy'], yerr=comparison_df['Accuracy_Std'],
                capsize=5, alpha=0.7, color='steelblue')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # MAE comparison
    axes[1].bar(models, comparison_df['MAE'], yerr=comparison_df['MAE_Std'],
                capsize=5, alpha=0.7, color='coral')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Model MAE Comparison')
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes