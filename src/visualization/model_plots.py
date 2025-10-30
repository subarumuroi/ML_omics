"""
Model-related visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(confusion_matrix, class_labels, title="Confusion Matrix", 
                          cmap="Blues", figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix (from sklearn.metrics.confusion_matrix)
    class_labels : list
        Class names
    title : str, default='Confusion Matrix'
        Plot title
    cmap : str, default='Blues'
        Colormap name
    figsize : tuple, default=(8, 6)
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_labels
    )
    disp.plot(cmap=cmap, values_format="d", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_feature_importance(importance_df, top_n=10, figsize=(10, 6), 
                           palette="viridis", title=None, save_path=None):
    """
    Plot feature importances as horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    top_n : int, default=10
        Number of top features to plot
    figsize : tuple, default=(10, 6)
        Figure size
    palette : str, default='viridis'
        Color palette
    title : str, optional
        Plot title (auto-generated if None)
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    data = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x="Importance", y="Feature", palette=palette, ax=ax)
    
    if title is None:
        title = f"Top {top_n} Features by Importance"
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_permutation_importance_boxplot(perm_result, feature_names, top_n=10, 
                                       figsize=(12, 6), save_path=None):
    """
    Plot distribution of permutation importances as boxplot.
    
    Shows the variability in importance across different permutations,
    providing insight into the stability of feature importance estimates.
    
    Parameters
    ----------
    perm_result : sklearn PermutationImportanceResult
        Result from sklearn.inspection.permutation_importance
    feature_names : list
        Names of features
    top_n : int, default=10
        Number of top features to plot
    figsize : tuple, default=(12, 6)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    # Get indices of top N features by mean importance
    top_indices = np.argsort(perm_result.importances_mean)[-top_n:][::-1]
    
    # Prepare data for boxplot
    box_data = [perm_result.importances[i] for i in top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(box_data, labels=top_feature_names, patch_artist=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xticklabels(top_feature_names, rotation=45, ha="right")
    ax.set_ylabel("Permutation Importance")
    ax.set_title(f"Permutation Importance Distribution (Top {top_n} Features)")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_permutation_test_distribution(perm_accuracies, observed_acc, 
                                       n_bins=30, figsize=(10, 6), save_path=None):
    """
    Plot permutation test null distribution.
    
    Shows the distribution of accuracies under the null hypothesis (no relationship
    between features and labels) with the observed accuracy marked.
    
    Parameters
    ----------
    perm_accuracies : np.ndarray
        Array of permutation test accuracies
    observed_acc : float
        Observed accuracy from real data
    n_bins : int, default=30
        Number of histogram bins
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram of permuted accuracies
    ax.hist(perm_accuracies, bins=n_bins, alpha=0.7, color="gray", 
            label="Permuted labels", edgecolor='black')
    
    # Mark observed accuracy
    ax.axvline(observed_acc, color="red", linestyle="--", linewidth=2,
               label=f"Observed accuracy ({observed_acc:.3f})")
    
    # Calculate p-value for display
    p_value = (np.sum(perm_accuracies >= observed_acc) + 1) / (len(perm_accuracies) + 1)
    
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Test Null Distribution (p = {p_value:.4f})")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_cv_scores(cv_scores, balanced_scores=None, figsize=(10, 6), save_path=None):
    """
    Plot cross-validation scores across folds.
    
    Parameters
    ----------
    cv_scores : list
        List of accuracy scores per fold
    balanced_scores : list, optional
        List of balanced accuracy scores per fold
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    folds = np.arange(1, len(cv_scores) + 1)
    
    ax.plot(folds, cv_scores, marker='o', label='Accuracy', linewidth=2)
    
    if balanced_scores is not None:
        ax.plot(folds, balanced_scores, marker='s', label='Balanced Accuracy', linewidth=2)
    
    # Add mean lines
    ax.axhline(np.mean(cv_scores), color='blue', linestyle='--', alpha=0.5,
               label=f'Mean Acc: {np.mean(cv_scores):.3f}')
    
    if balanced_scores is not None:
        ax.axhline(np.mean(balanced_scores), color='orange', linestyle='--', alpha=0.5,
                   label=f'Mean Bal Acc: {np.mean(balanced_scores):.3f}')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Scores')
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax