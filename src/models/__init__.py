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
