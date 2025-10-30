# ============================================================================
# src/visualization/__init__.py
# ============================================================================
"""
Visualization module for plotting model results and data exploration.
"""

from .model_plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance_boxplot,
    plot_permutation_test_distribution,
    plot_cv_scores,
)
from .data_plots import (
    plot_compound_trends,
    plot_compound_boxplots,
    plot_compound_heatmap,
    plot_pairwise_compounds,
    plot_correlation_heatmap,
    plot_violin_by_group,
)
from .dimension_reduction import (
    plot_tsne,
    plot_umap,
    plot_shap_summary,
    plot_shap_waterfall,
)

# Ordinal-specific plots (optional)
try:
    from .ordinal_plots import (
        plot_ordinal_coefficients,
        plot_ordinal_confusion_matrix,
        plot_ordinal_predictions_vs_true,
        plot_ordinal_cv_performance,
        plot_ordinal_model_comparison,
    )
    ORDINAL_PLOTS_AVAILABLE = True
except ImportError:
    ORDINAL_PLOTS_AVAILABLE = False

__all__ = [
    # Model plots
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_permutation_importance_boxplot',
    'plot_permutation_test_distribution',
    'plot_cv_scores',
    # Data plots
    'plot_compound_trends',
    'plot_compound_boxplots',
    'plot_compound_heatmap',
    'plot_pairwise_compounds',
    'plot_correlation_heatmap',
    'plot_violin_by_group',
    # Dimension reduction
    'plot_tsne',
    'plot_umap',
    'plot_shap_summary',
    'plot_shap_waterfall',
]

if ORDINAL_PLOTS_AVAILABLE:
    __all__.extend([
        'plot_ordinal_coefficients',
        'plot_ordinal_confusion_matrix',
        'plot_ordinal_predictions_vs_true',
        'plot_ordinal_cv_performance',
        'plot_ordinal_model_comparison',
    ])
