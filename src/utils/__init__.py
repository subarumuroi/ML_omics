
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