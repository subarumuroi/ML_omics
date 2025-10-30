"""
Ordinal regression workflow example.

Demonstrates ordinal regression for ordered categorical outcomes
(e.g., Green < Ripe < Overripe) and integration with other package modules.

Requires: pip install mord
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import package modules
from preprocessing import load_and_impute
from models.ordinal import (
    OrdinalClassifier,
    train_evaluate_ordinal,
    compare_ordinal_models,
    encode_ordinal_target,
)
from models import train_evaluate_model  # For comparison with RF
from utils import (
    prepare_data,
    set_categorical_order,
    print_data_summary,
    create_results_directory,
    save_results,
)
from visualization import (
    plot_compound_trends,
    plot_compound_boxplots,
)
from visualization.ordinal_plots import (
    plot_ordinal_coefficients,
    plot_ordinal_confusion_matrix,
    plot_ordinal_predictions_vs_true,
    plot_ordinal_cv_performance,
    plot_ordinal_model_comparison,
)
from visualization.data_plots import (
    plot_correlation_heatmap,
    plot_violin_by_group,
)


def main():
    """Run complete ordinal regression workflow."""
    
    # ======================================================================
    # CONFIGURATION
    # ======================================================================
    DATA_FILE = "badata_imputed.csv"
    GROUP_COL = "Groups"
    GROUP_ORDER = ["Green", "Ripe", "Overripe"]
    TOP_N_FEATURES = 10
    
    print("\n" + "="*70)
    print("ORDINAL REGRESSION WORKFLOW")
    print("="*70 + "\n")
    
    # Create results directory
    results_dirs = create_results_directory(
        base_path='results',
        run_name='ordinal_analysis'
    )
    
    # ======================================================================
    # 1. LOAD AND PREPARE DATA
    # ======================================================================
    print("STEP 1: Loading and preparing data...")
    
    df = load_and_impute(DATA_FILE, group_col=GROUP_COL, impute=True, fill_value=0)
    df = set_categorical_order(df, col=GROUP_COL, categories=GROUP_ORDER)
    
    # Prepare data
    X, y_raw, _, _, _ = prepare_data(df, target_col=GROUP_COL, drop_missing=True)
    
    # Encode as ordinal (0=Green, 1=Ripe, 2=Overripe)
    y = encode_ordinal_target(y_raw, categories=GROUP_ORDER)
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes: {GROUP_ORDER}")
    print(f"Encoding: {dict(zip(GROUP_ORDER, range(len(GROUP_ORDER))))}")
    
    # ======================================================================
    # 2. TRAIN ORDINAL REGRESSION MODEL
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 2: Training Ordinal Regression Model")
    print("="*70)
    
    # Train with cross-validation
    results = train_evaluate_ordinal(
        X, y,
        model_type='LogisticAT',
        scaler='minmax',
        log_transform=True,
        n_splits=None,
        verbose=True
    )
    
    # Extract results
    model = results['model']
    coef_df = results['coefficients']
    cv_scores = results['cv_scores']
    cv_mae = results['cv_mae']
    
    print(f"\nTop {TOP_N_FEATURES} important features:")
    print(coef_df.head(TOP_N_FEATURES))
    
    # ======================================================================
    # 3. VISUALIZE COEFFICIENTS
    # ======================================================================
    print("\nSTEP 3: Visualizing model coefficients...")
    
    fig, _ = plot_ordinal_coefficients(coef_df, top_n=TOP_N_FEATURES)
    fig.savefig(results_dirs['figures'] / 'ordinal_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 4. CROSS-VALIDATION PERFORMANCE
    # ======================================================================
    print("\nSTEP 4: Cross-validation performance...")
    
    fig, _ = plot_ordinal_cv_performance(cv_scores, cv_mae)
    fig.savefig(results_dirs['figures'] / 'ordinal_cv_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 5. CONFUSION MATRIX
    # ======================================================================
    print("\nSTEP 5: Generating confusion matrix...")
    
    fig, _ = plot_ordinal_confusion_matrix(
        results['all_true'],
        results['all_pred'],
        class_labels=GROUP_ORDER
    )
    fig.savefig(results_dirs['figures'] / 'ordinal_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Predictions vs true
    fig, _ = plot_ordinal_predictions_vs_true(
        results['all_true'],
        results['all_pred'],
        class_labels=GROUP_ORDER
    )
    fig.savefig(results_dirs['figures'] / 'ordinal_pred_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 6. COMPOUND TRENDS (TOP FEATURES)
    # ======================================================================
    print("\nSTEP 6: Visualizing compound trends...")
    
    top_features = coef_df.head(TOP_N_FEATURES)['Feature'].tolist()
    
    # Linear scale
    fig, _ = plot_compound_trends(
        df, top_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER,
        scale='linear'
    )
    fig.savefig(results_dirs['figures'] / 'compound_trends_linear.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log scale
    fig, _ = plot_compound_trends(
        df, top_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER,
        scale='log'
    )
    fig.savefig(results_dirs['figures'] / 'compound_trends_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplots
    fig, _ = plot_compound_boxplots(
        df, top_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER
    )
    fig.savefig(results_dirs['figures'] / 'compound_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 7. CORRELATION HEATMAP
    # ======================================================================
    print("\nSTEP 7: Correlation analysis...")
    
    fig, _ = plot_correlation_heatmap(df, top_features)
    fig.savefig(results_dirs['figures'] / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 8. VIOLIN PLOTS
    # ======================================================================
    print("\nSTEP 8: Distribution analysis...")
    
    fig, _ = plot_violin_by_group(
        df, top_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER
    )
    fig.savefig(results_dirs['figures'] / 'violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 9. COMPARE ORDINAL MODELS
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 9: Comparing Ordinal Models")
    print("="*70)
    
    comparison = compare_ordinal_models(
        X, y,
        model_types=['LogisticAT', 'LogisticIT', 'LogisticSE'],
        verbose=True
    )
    
    fig, _ = plot_ordinal_model_comparison(comparison)
    fig.savefig(results_dirs['figures'] / 'ordinal_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 10. COMPARE WITH RANDOM FOREST
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 10: Comparing with Random Forest")
    print("="*70)
    
    # Train RF for comparison
    from preprocessing import select_k_best_features
    X_selected, selected_features, _ = select_k_best_features(X, y, k=15, verbose=False)
    X_df = pd.DataFrame(X_selected, columns=selected_features)
    
    rf_clf = train_evaluate_model(X_df, y, verbose=True)
    
    # Print comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\nOrdinal Regression (LogisticAT):")
    print(f"  Accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    print(f"  MAE: {results['mean_mae']:.3f} ± {results['std_mae']:.3f}")
    print(f"\nRandom Forest:")
    print(f"  Accuracy: {rf_clf.cv_results['mean_accuracy']:.3f} ± {rf_clf.cv_results['std_accuracy']:.3f}")
    print(f"  Balanced Accuracy: {rf_clf.cv_results['mean_balanced_accuracy']:.3f}")
    
    # ======================================================================
    # 11. SAVE RESULTS
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 11: Saving Results")
    print("="*70)
    
    # Save coefficients
    coef_df.to_csv(results_dirs['data'] / 'ordinal_coefficients.csv', index=False)
    
    # Save comparison
    comparison.to_csv(results_dirs['data'] / 'ordinal_model_comparison.csv', index=False)
    
    # Save summary
    summary = {
        'model_type': 'LogisticAT',
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'cv_accuracy': results['mean_accuracy'],
        'cv_accuracy_std': results['std_accuracy'],
        'cv_mae': results['mean_mae'],
        'cv_mae_std': results['std_mae'],
        'top_features': top_features,
        'ordinal_encoding': dict(zip(GROUP_ORDER, range(len(GROUP_ORDER)))),
    }
    
    save_results(summary, results_dirs['base'] / 'ordinal_results.json')
    
    print(f"\nResults saved to: {results_dirs['base']}")
    
    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOrdinal Regression Performance:")
    print(f"  Accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    print(f"  MAE: {results['mean_mae']:.3f}")
    print(f"\nTop 3 features predicting ripeness:")
    for i, row in coef_df.head(3).iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {i+1}. {row['Feature']}: {direction} ripeness (coef={row['Coefficient']:.3f})")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()