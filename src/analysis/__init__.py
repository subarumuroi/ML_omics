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