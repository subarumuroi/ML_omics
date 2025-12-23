# Add after your imports at top
"""
Network and module analysis for metabolomics and proteomics data.

Functions for identifying metabolite and/or proteome clusters and modules based on 
correlation patterns, useful for expanding from minimal biomarker 
sets to broader biological context.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_network_modules(df, key_features, feature_col_prefix='', 
                               corr_threshold=0.7, distance_metric='correlation',
                               linkage_method='average', cut_height=0.3):
    """
    Identify metabolite modules around key features using hierarchical clustering.
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset with all metabolites
    key_features : list
        List of key feature names (e.g., ['48_SUCC_0.0', '47_SHIK 0.0'])
    corr_threshold : float
        Minimum |correlation| to consider metabolites related
    distance_metric : str
        'correlation' uses 1-|r| as distance
    linkage_method : str
        'average', 'complete', 'ward', etc.
    cut_height : float
        Height to cut dendrogram tree
        
    Returns:
    --------
    dict : modules dictionary with structure:
        {feature_name: {
            'correlated_features': list,
            'correlation_values': dict,
            'cluster_members': list,
            'cluster_id': int
        }}
    """
    
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    modules = {}
    
    for key_feature in key_features:
        if key_feature not in corr_matrix.columns:
            print(f"Warning: {key_feature} not in correlation matrix")
            continue
            
        # Find highly correlated metabolites
        correlations = corr_matrix[key_feature].abs()
        related_features = correlations[correlations >= corr_threshold].sort_values(ascending=False)
        
        # Get the subset for clustering
        cluster_features = related_features.index.tolist()
        
        if len(cluster_features) < 2:
            print(f"Warning: Only {len(cluster_features)} features correlated with {key_feature}")
            modules[key_feature] = {
                'correlated_features': cluster_features,
                'correlation_values': related_features.to_dict(),
                'cluster_members': cluster_features,
                'cluster_id': 0
            }
            continue
        
        # Hierarchical clustering on correlated subset
        subset_corr = corr_matrix.loc[cluster_features, cluster_features]
        
        # Convert correlation to distance: d = 1 - |r|
        distance_matrix = 1 - subset_corr.abs()
        
        # Perform hierarchical clustering
        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method=linkage_method)
        
        # Cut tree to get clusters
        clusters = fcluster(Z, t=cut_height, criterion='distance')
        
        # Find which cluster contains the key feature
        key_cluster_id = clusters[cluster_features.index(key_feature)]
        cluster_members = [f for f, c in zip(cluster_features, clusters) if c == key_cluster_id]
        
        modules[key_feature] = {
            'correlated_features': cluster_features,
            'correlation_values': related_features.to_dict(),
            'cluster_members': cluster_members,
            'cluster_id': key_cluster_id,
            'all_clusters': dict(zip(cluster_features, clusters)),
            'linkage_matrix': Z
        }
        
        print(f"\n{key_feature} module:")
        print(f"  - {len(related_features)} metabolites correlated (r > {corr_threshold})")
        print(f"  - {len(cluster_members)} in core cluster")
        
    return modules, corr_matrix



# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_network(modules, corr_matrix, key_feature, results_dir, 
                            figsize=(12, 10)):
    """
    Visualize metabolite module as heatmap and dendrogram.
    """
    if key_feature not in modules:
        return None
        
    module = modules[key_feature]
    cluster_features = module['correlated_features']
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Dendrogram
    ax_dend = plt.subplot(2, 1, 1)
    Z = module['linkage_matrix']
    dendrogram(Z, labels=cluster_features, ax=ax_dend)
    ax_dend.set_title(f'Hierarchical Clustering: {key_feature} Module')
    ax_dend.set_xlabel('Metabolite')
    ax_dend.set_ylabel('Distance (1-|r|)')
    plt.xticks(rotation=90)
    
    # Correlation heatmap
    ax_heat = plt.subplot(2, 1, 2)
    subset_corr = corr_matrix.loc[cluster_features, cluster_features]
    sns.heatmap(subset_corr, annot=False, cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, ax=ax_heat, cbar_kws={'label': 'Correlation'})
    ax_heat.set_title(f'Correlation Matrix: {key_feature} Module')
    
    plt.tight_layout()
    
    return fig