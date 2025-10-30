"""
Data exploration and compound analysis visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_compound_trends(df, feature_names, group_col='Groups', 
                        group_order=None, scale='linear', figsize=(12, 6),
                        save_path=None):
    """
    Plot compound/feature trends across groups as line plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with groups and features
    feature_names : list
        Names of features/compounds to plot
    group_col : str, default='Groups'
        Column name for grouping
    group_order : list, optional
        Order of groups for x-axis (e.g., ['Green', 'Ripe', 'Overripe'])
    scale : str, default='linear'
        Y-axis scale ('linear' or 'log')
    figsize : tuple, default=(12, 6)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    # Compute group means
    grouped_means = df.groupby(group_col)[feature_names].mean().reset_index()
    
    # Melt for line plot
    melted_df = grouped_means.melt(
        id_vars=group_col,
        var_name='Compound',
        value_name='Mean Area'
    )
    
    # Set categorical order if provided
    if group_order is not None:
        melted_df[group_col] = pd.Categorical(
            melted_df[group_col],
            categories=group_order,
            ordered=True
        )
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=melted_df,
        x=group_col,
        y='Mean Area',
        hue='Compound',
        marker='o',
        ax=ax,
        linewidth=2,
        markersize=8
    )
    
    if scale == 'log':
        ax.set_yscale('log')
        title_scale = " (Log Scale)"
    else:
        title_scale = " (Linear Scale)"
    
    ax.set_title(f"Compound Trends Across Groups{title_scale}")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Mean Area")
    ax.grid(True, alpha=0.3)
    
    # Adjust legend position for log scale
    if scale == 'log':
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(feature_names)),
            title="Compound",
            frameon=False
        )
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_compound_boxplots(df, feature_names, group_col='Groups', 
                           group_order=None, figsize=(12, 6), save_path=None):
    """
    Plot compound/feature distributions as boxplot by group.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with groups and features
    feature_names : list
        Names of features/compounds to plot
    group_col : str, default='Groups'
        Column name for grouping
    group_order : list, optional
        Order of groups
    figsize : tuple, default=(12, 6)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    # Prepare data
    df_plot = df[[group_col] + list(feature_names)]
    df_melted = df_plot.melt(
        id_vars=group_col,
        var_name='Compound',
        value_name='Area'
    )
    
    # Set categorical order if provided
    if group_order is not None:
        df_melted[group_col] = pd.Categorical(
            df_melted[group_col],
            categories=group_order,
            ordered=True
        )
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df_melted,
        x='Compound',
        y='Area',
        hue=group_col,
        ax=ax
    )
    
    ax.set_title(f"Compound Distribution by {group_col}")
    ax.set_xlabel("Compound")
    ax.set_ylabel("Area")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_compound_heatmap(df, feature_names, group_col='Groups',
                         group_order=None, figsize=(10, 8), cmap='viridis',
                         save_path=None):
    """
    Plot heatmap of compound means across groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with groups and features
    feature_names : list
        Names of features/compounds to plot
    group_col : str, default='Groups'
        Column name for grouping
    group_order : list, optional
        Order of groups
    figsize : tuple, default=(10, 8)
        Figure size
    cmap : str, default='viridis'
        Colormap
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    # Compute group means
    grouped_means = df.groupby(group_col)[feature_names].mean()
    
    # Reorder if specified
    if group_order is not None:
        grouped_means = grouped_means.reindex(group_order)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        grouped_means.T,
        annot=True,
        fmt='.2e',
        cmap=cmap,
        ax=ax,
        cbar_kws={'label': 'Mean Area'}
    )
    
    ax.set_title("Compound Heatmap Across Groups")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Compound")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_pairwise_compounds(df, feature_names, group_col='Groups', 
                           group_order=None, figsize=(12, 10), save_path=None):
    """
    Create pairplot for selected compounds colored by group.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with groups and features
    feature_names : list
        Names of features/compounds to plot (max 5-6 recommended)
    group_col : str, default='Groups'
        Column name for grouping
    group_order : list, optional
        Order of groups
    figsize : tuple, default=(12, 10)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    sns.PairGrid
        Seaborn PairGrid object
    """
    # Prepare data
    plot_data = df[[group_col] + list(feature_names)].copy()
    
    # Set categorical order if provided
    if group_order is not None:
        plot_data[group_col] = pd.Categorical(
            plot_data[group_col],
            categories=group_order,
            ordered=True
        )
    
    # Create pairplot
    g = sns.pairplot(
        plot_data,
        hue=group_col,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 80},
        height=2.5
    )
    
    g.fig.suptitle("Pairwise Compound Relationships", y=1.01)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return g


def plot_correlation_heatmap(df, feature_names, figsize=(12, 10), 
                             annot=True, fmt=".2f", cmap='coolwarm',
                             save_path=None):
    """
    Plot correlation heatmap of selected features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
    feature_names : list
        Names of features to include
    figsize : tuple, default=(12, 10)
        Figure size
    annot : bool, default=True
        Annotate cells with correlation values
    fmt : str, default='.2f'
        Format string for annotations
    cmap : str, default='coolwarm'
        Colormap
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    corr_matrix = df[feature_names].corr()
    sns.heatmap(
        corr_matrix, 
        annot=annot, 
        cmap=cmap, 
        fmt=fmt, 
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title(f'Correlation Heatmap of Top {len(feature_names)} Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_violin_by_group(df, feature_names, group_col='Groups', 
                        group_order=None, figsize=(14, 10), save_path=None):
    """
    Plot violin plots of features across groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with groups and features
    feature_names : list
        Names of features to plot
    group_col : str, default='Groups'
        Column name for grouping
    group_order : list, optional
        Order of groups
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes
    """
    # Melt data
    melted = df.melt(
        id_vars=group_col,
        value_vars=feature_names,
        var_name='Compound',
        value_name='Area'
    )
    
    # Set categorical order if provided
    if group_order is not None:
        melted[group_col] = pd.Categorical(
            melted[group_col],
            categories=group_order,
            ordered=True
        )
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=melted,
        x='Compound',
        y='Area',
        hue=group_col,
        split=False,
        ax=ax
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(f'Distribution of Top {len(feature_names)} Features Across Groups')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax