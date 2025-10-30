# ============================================================================
# src/__init__.py
# ============================================================================
"""
ML Analysis Package

A modular package for machine learning analysis with Random Forest 
classification, feature selection, permutation testing, and visualization.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# ============================================================================
# src/preprocessing/__init__.py
# ============================================================================
"""
Preprocessing module for data loading and feature selection.
"""

from .data_processing import load_and_impute
from .feature_selection import (
    select_k_best_features,
    get_feature_scores,
    prepare_feature_subset,
    get_top_n_features,
)

__all__ = [
    'load_and_impute',
    'select_k_best_features',
    'get_feature_scores',
    'prepare_feature_subset',
    'get_top_n_features',
]

# ============================================================================
# src/models/__init__.py
# ============================================================================
"""
Models module for Random Forest classification, ordinal regression, and evaluation.
"""

from .random_forest import RFClassifierCV, train_evaluate_model
from .evaluation import (
    get_classification_report,
    get_confusion_matrix,
    compute_cv_metrics,
    create_results_summary,
    compare_models,
)

# Ordinal regression (optional - requires mord)
try:
    from .ordinal import (
        OrdinalClassifier,
        train_evaluate_ordinal,
        compare_ordinal_models,
        encode_ordinal_target,
    )
    ORDINAL_AVAILABLE = True
except ImportError:
    ORDINAL_AVAILABLE = False

__all__ = [
    'RFClassifierCV',
    'train_evaluate_model',
    'get_classification_report',
    'get_confusion_matrix',
    'compute_cv_metrics',
    'create_results_summary',
    'compare_models',
]

if ORDINAL_AVAILABLE:
    __all__.extend([
        'OrdinalClassifier',
        'train_evaluate_ordinal',
        'compare_ordinal_models',
        'encode_ordinal_target',
    ])

# ============================================================================
# src/analysis/__init__.py
# ============================================================================
"""
Analysis module for feature importance and statistical testing.
"""

from .importance import (
    get_feature_importance_df,
    compute_permutation_importance,
    compare_importance_methods,
    get_shap_values,
    get_shap_importance,
)
from .statistical_tests import (
    permutation_test,
    permutation_test_with_feature_subset,
    compare_permutation_tests,
)

__all__ = [
    'get_feature_importance_df',
    'compute_permutation_importance',
    'compare_importance_methods',
    'get_shap_values',
    'get_shap_importance',
    'permutation_test',
    'permutation_test_with_feature_subset',
    'compare_permutation_tests',
]

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

# ============================================================================
# src/utils/__init__.py
# ============================================================================
"""
Utilities module for data preparation and I/O operations.
"""

from .data_utils import (
    prepare_data,
    set_categorical_order,
    get_cv_splits,
    print_data_summary,
    check_class_balance,
    drop_index_columns,
    remove_low_variance_features,
)
from .io_utils import (
    save_results,
    load_results,
    save_dataframe,
    save_figure,
    create_results_directory,
    save_model,
    load_model,
)

__all__ = [
    # Data utils
    'prepare_data',
    'set_categorical_order',
    'get_cv_splits',
    'print_data_summary',
    'check_class_balance',
    'drop_index_columns',
    'remove_low_variance_features',
    # IO utils
    'save_results',
    'load_results',
    'save_dataframe',
    'save_figure',
    'create_results_directory',
    'save_model',
    'load_model',
]