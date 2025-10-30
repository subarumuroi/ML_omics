"""
Dimensionality reduction and interpretability visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# UMAP is optional
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def plot_tsne(X, y, label_encoder=None, figsize=(8, 6), perplexity=5, 
              random_state=42, title="t-SNE Visualization", save_path=None):
    """
    Create t-SNE visualization of samples.
    
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is useful for visualizing
    high-dimensional data in 2D by preserving local structure.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels (encoded as integers)
    label_encoder : sklearn.LabelEncoder, optional
        Encoder to get class names from encoded labels
    figsize : tuple, default=(8, 6)
        Figure size
    perplexity : int, default=5
        t-SNE perplexity parameter (roughly the number of nearest neighbors)
    random_state : int, default=42
        Random state for reproducibility
    title : str, default='t-SNE Visualization'
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax, X_tsne) matplotlib figure, axes, and transformed coordinates
    """
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=1000
    )
    X_tsne = tsne.fit_transform(X)
    
    # Get labels
    if label_encoder is not None:
        labels = label_encoder.inverse_transform(y)
    else:
        labels = y
    
    fig, ax = plt.subplots(figsize=figsize)
    scatter = sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=labels,
        palette="Set1",
        s=100,
        alpha=0.7,
        ax=ax,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax, X_tsne


def plot_umap(X, y, label_encoder=None, figsize=(8, 6), n_neighbors=10, 
              min_dist=0.1, random_state=42, title="UMAP Visualization", save_path=None):
    """
    Create UMAP visualization of samples.
    
    UMAP (Uniform Manifold Approximation and Projection) is often faster than t-SNE
    and can preserve more global structure.
    
    Requires: pip install umap-learn
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels (encoded as integers)
    label_encoder : sklearn.LabelEncoder, optional
        Encoder to get class names
    figsize : tuple, default=(8, 6)
        Figure size
    n_neighbors : int, default=10
        UMAP n_neighbors parameter (controls local vs global structure)
    min_dist : float, default=0.1
        UMAP min_dist parameter (controls clustering tightness)
    random_state : int, default=42
        Random state
    title : str, default='UMAP Visualization'
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax, X_umap) matplotlib figure, axes, and transformed coordinates
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    umap_model = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )
    X_umap = umap_model.fit_transform(X)
    
    # Get labels
    if label_encoder is not None:
        labels = label_encoder.inverse_transform(y)
    else:
        labels = y
    
    fig, ax = plt.subplots(figsize=figsize)
    scatter = sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=labels,
        palette="Set1",
        s=100,
        alpha=0.7,
        ax=ax,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax, X_umap


def plot_shap_summary(shap_values, X, class_labels=None, max_display=10, save_path=None):
    """
    Plot SHAP summary plots for each class.
    
    SHAP (SHapley Additive exPlanations) values show how much each feature
    contributes to pushing the model output from the base value.
    
    Requires: pip install shap
    
    Parameters
    ----------
    shap_values : list or np.ndarray
        SHAP values from TreeExplainer
    X : pd.DataFrame
        Feature matrix
    class_labels : list, optional
        Class names
    max_display : int, default=10
        Maximum features to display
    save_path : str, optional
        Base path to save figures (will append class name)
        
    Notes
    -----
    For multiclass problems, creates one plot per class.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(len(shap_values))]
    
    # Handle multiclass (list of arrays)
    if isinstance(shap_values, list):
        for class_idx, class_label in enumerate(class_labels):
            print(f"\nSHAP summary for: {class_label}")
            shap_class = shap_values[class_idx]
            
            plt.figure()
            shap.summary_plot(shap_class, X, max_display=max_display, show=False)
            plt.title(f"SHAP Summary - {class_label}")
            plt.tight_layout()
            
            if save_path:
                path = save_path.replace('.png', f'_{class_label}.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
            
            plt.show()
    else:
        # Binary classification
        print(f"\nSHAP summary")
        plt.figure()
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title("SHAP Summary")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def plot_shap_waterfall(shap_values, X, sample_idx, class_idx=None, class_label=None,
                       max_display=10, save_path=None):
    """
    Plot SHAP waterfall plot for a single sample.
    
    Shows how each feature contributes to the prediction for a specific sample.
    
    Parameters
    ----------
    shap_values : list or np.ndarray
        SHAP values from TreeExplainer
    X : pd.DataFrame
        Feature matrix
    sample_idx : int
        Index of sample to explain
    class_idx : int, optional
        For multiclass, which class to show
    class_label : str, optional
        Label for the class
    max_display : int, default=10
        Maximum features to display
    save_path : str, optional
        Path to save figure
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    # Handle multiclass
    if isinstance(shap_values, list):
        if class_idx is None:
            raise ValueError("For multiclass, must specify class_idx")
        shap_vals = shap_values[class_idx][sample_idx]
    else:
        shap_vals = shap_values[sample_idx]
    
    title = f"SHAP Waterfall - Sample {sample_idx}"
    if class_label:
        title += f" ({class_label})"
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=0,  # Simplified
            data=X.iloc[sample_idx] if hasattr(X, 'iloc') else X[sample_idx],
            feature_names=X.columns if hasattr(X, 'columns') else None
        ),
        max_display=max_display,
        show=False
    )
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()