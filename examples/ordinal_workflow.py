
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
from preprocessing import (
    load_and_impute,
    select_k_best_features,
    get_feature_scores,
)
from models.ordinal import (
    OrdinalClassifier,
    train_evaluate_ordinal,
    compare_ordinal_models,
    encode_ordinal_target,
)
from models import train_evaluate_rf  # For comparison with RF
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
    
    # Configuration
    DATA_FILE = "data/badata-proteomics-unimputed.csv"
    GROUP_COL = "Groups"
    GROUP_ORDER = ["Green", "Ripe", "Overripe"]
    TOP_N_FEATURES = 10
    TOP_N_FEATURES_FOR_UNIVARIATE = 10
    
    print("\n" + "="*70)
    print("ORDINAL REGRESSION WORKFLOW")
    print("="*70 + "\n")
    
    results_dirs = create_results_directory(base_path='results', run_name='ordinal_analysis')
    
    # Load and prepare data
    print("STEP 1: Loading and preparing data...")
    df = load_and_impute(DATA_FILE, group_col=GROUP_COL, impute=True, fill_value=0, drop_threshold=0.35)
    df = df.drop(columns='Unnamed: 0', errors='ignore')  # Drop index column
    df = set_categorical_order(df, col=GROUP_COL, categories=GROUP_ORDER)
    X, y_raw, _, _, _ = prepare_data(df, target_col=GROUP_COL, drop_missing=True)
    X = X.select_dtypes(include=[np.number])
    
    #=== fixes problem of negative values in metab ===#
    # Handle negative values for metabolomics data (before log transformation)
    if (X < 0).any().any():
        n_negatives = (X < 0).sum().sum()
        print(f"\nWarning: Found {n_negatives} negative values across {(X < 0).any().sum()} features")
        print("Applying per-feature shifting to positive range for log transformation...")
        
        # Per-feature shift: each negative feature shifted independently
        for col in X.columns:
            if (X[col] < 0).any():
                shift_amount = abs(X[col].min()) + 1
                X[col] = X[col] + shift_amount
        
        print(f"✓ Shifted {(X < 0).any().sum()} features independently")
    #=== end fix ===#

    print(f"Initial features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Get top N univariate features with scores and p-values

    X_full = X.copy()
    scores_df = get_feature_scores(X_full, y_raw)  # Returns Feature, Score, P_value

    # Take top N features
    top_features_all = scores_df['Feature'].head(TOP_N_FEATURES_FOR_UNIVARIATE).tolist()

    # Print to terminal for transparency
    print(f"\nTop {TOP_N_FEATURES_FOR_UNIVARIATE} features by univariate score (F-statistic) and p-value:")
    for i, row in scores_df.head(TOP_N_FEATURES_FOR_UNIVARIATE).iterrows():
        print(f"  {i+1}. {row['Feature']}: Score={row['Score']:.3f}, P-value={row['P_value']:.4g}")

    # Save CSV for scientific reporting
    scores_df.head(TOP_N_FEATURES_FOR_UNIVARIATE).to_csv(
        results_dirs['data'] / f'top{TOP_N_FEATURES_FOR_UNIVARIATE}_univariate_scores.csv',
        index=False
    )

    print(f"\nGenerating extended compound plots for top {TOP_N_FEATURES_FOR_UNIVARIATE} features...")


    for scale in ['linear', 'log']:
        fig, _ = plot_compound_trends(df, top_features_all, group_col=GROUP_COL,
                                    group_order=GROUP_ORDER, scale=scale)
        fig.savefig(results_dirs['figures'] / f'compound_trends_top{TOP_N_FEATURES_FOR_UNIVARIATE}_{scale}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    fig, _ = plot_compound_boxplots(df, top_features_all, group_col=GROUP_COL, group_order=GROUP_ORDER)
    fig.savefig(results_dirs['figures'] / f'compound_boxplots_top{TOP_N_FEATURES_FOR_UNIVARIATE}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # STEP 1.5: Determine optimal number of features
    print("\n" + "="*70)
    print("STEP 1.5: Feature Selection Analysis")
    print("="*70)
    
    feature_counts = [1, 2, 3, 5, 7, 10, 15]
    results_by_k = {}
    
    for k in feature_counts:
        if k > X.shape[1]:
            print(f"Skipping k={k} (exceeds available features)")
            continue
        
        X_k, features_k, _ = select_k_best_features(X, y_raw, k=k, verbose=False)
        y_encoded = np.array(encode_ordinal_target(y_raw, categories=GROUP_ORDER))
        
        results = train_evaluate_ordinal(X_k, y_encoded, model_type='LogisticAT', 
                                         scaler='minmax', log_transform=True, 
                                         n_splits=None, verbose=False)
        results_by_k[k] = {
            'accuracy': results['mean_accuracy'],
            'mae': results['mean_mae'],
            'std_accuracy': results['std_accuracy'],
            'features': list(features_k)
        }
        print(f"k={k:2d}: Accuracy={results['mean_accuracy']:.3f}±{results['std_accuracy']:.3f}, MAE={results['mean_mae']:.3f}")
    
    # Select optimal k (prioritize lower MAE (better ordinal predictions),
    # then higher accuracy as tiebreaker
    optimal_k = min(results_by_k.keys(), key=lambda k: (results_by_k[k]['mae'], -results_by_k[k]['accuracy']))
    print(f"\n✓ Optimal number of features: {optimal_k}")
    print(f"  Selected features: {results_by_k[optimal_k]['features']}")

    # Use optimal k for final model
    X, selected_features, _ = select_k_best_features(X, y_raw, k=optimal_k, verbose=False)
    X = pd.DataFrame(X, columns=selected_features)
    y = np.array(encode_ordinal_target(y_raw, categories=GROUP_ORDER))
    
    print(f"\nFinal feature set: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"Encoding: {dict(zip(GROUP_ORDER, range(len(GROUP_ORDER))))}")
    
    # Train ordinal regression model
    print("\n" + "="*70)
    print("STEP 2: Training Ordinal Regression Model")
    print("="*70)
    
    results = train_evaluate_ordinal(X, y, model_type='LogisticAT', scaler='minmax', 
                                     log_transform=True, n_splits=None, verbose=True)
    
    model = results['model']
    coef_df = results['coefficients']
    
    print(f"\nModel coefficients (top {min(TOP_N_FEATURES, len(coef_df))} features by |β|):")

    for i, row in coef_df.head(TOP_N_FEATURES).iterrows():
        direction = "↑ increases" if row['Coefficient'] > 0 else "↓ decreases"
        print(f"  {i+1}. {row['Feature']}: {direction} ripeness (β={row['Coefficient']:.3f})")
        
    # Visualizations
    print("\nSTEP 3-5: Generating visualizations...")

    # Coefficients
    fig, _ = plot_ordinal_coefficients(coef_df, top_n=min(TOP_N_FEATURES, len(coef_df)))
    fig.savefig(results_dirs['figures'] / 'ordinal_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # CV performance
    fig, _ = plot_ordinal_cv_performance(results['cv_scores'], results['cv_mae'])
    fig.savefig(results_dirs['figures'] / 'ordinal_cv_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    fig, _ = plot_ordinal_confusion_matrix(results['all_true'], results['all_pred'], class_labels=GROUP_ORDER)
    fig.savefig(results_dirs['figures'] / 'ordinal_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Predictions vs true
    fig, _ = plot_ordinal_predictions_vs_true(results['all_true'], results['all_pred'], class_labels=GROUP_ORDER)
    fig.savefig(results_dirs['figures'] / 'ordinal_pred_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compound analysis
    print("\nSTEP 6-8: Compound analysis...")
    
    top_features = coef_df['Feature'].tolist()  # Use all selected features
    
    for scale in ['linear', 'log']:
        fig, _ = plot_compound_trends(df, top_features, group_col=GROUP_COL, 
                                      group_order=GROUP_ORDER, scale=scale)
        fig.savefig(results_dirs['figures'] / f'compound_trends_{scale}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    fig, _ = plot_compound_boxplots(df, top_features, group_col=GROUP_COL, group_order=GROUP_ORDER)
    fig.savefig(results_dirs['figures'] / 'compound_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, _ = plot_correlation_heatmap(df, top_features)
    fig.savefig(results_dirs['figures'] / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, _ = plot_violin_by_group(df, top_features, group_col=GROUP_COL, group_order=GROUP_ORDER)
    fig.savefig(results_dirs['figures'] / 'violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Compare ordinal models
    print("\n" + "="*70)
    print("STEP 9: Comparing Ordinal Models")
    print("="*70)
    
    comparison = compare_ordinal_models(X, y, model_types=['LogisticAT', 'LogisticIT', 'LogisticSE'], verbose=True)
    
    fig, _ = plot_ordinal_model_comparison(comparison)
    fig.savefig(results_dirs['figures'] / 'ordinal_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare with Random Forest
    print("\n" + "="*70)
    print("STEP 10: Comparing with Random Forest")
    print("="*70)
    
    rf_clf = train_evaluate_rf(X, y, verbose=True)

    # Save results
    print("\n" + "="*70)
    print("STEP 11: Saving Results")
    print("="*70)
    
    coef_df.to_csv(results_dirs['data'] / 'ordinal_coefficients.csv', index=False)
    comparison.to_csv(results_dirs['data'] / 'ordinal_model_comparison.csv', index=False)
    
    # Save feature selection results
    feature_selection_df = pd.DataFrame([
        {'k': k, 'accuracy': v['accuracy'], 'std_accuracy': v['std_accuracy'], 
         'mae': v['mae'], 'features': ', '.join(v['features'])}
        for k, v in results_by_k.items()
    ])
    feature_selection_df.to_csv(results_dirs['data'] / 'feature_selection_analysis.csv', index=False)
    
    summary = {
        'model_type': 'LogisticAT',
        'optimal_k': optimal_k,
        'n_features': X.shape[1],
        'cv_accuracy': results['mean_accuracy'],
        'cv_mae': results['mean_mae'],
        'selected_features': list(selected_features),
        'feature_selection_results': results_by_k,
        'ordinal_encoding': dict(zip(GROUP_ORDER, range(len(GROUP_ORDER)))),
    }
    
    save_results(summary, results_dirs['base'] / 'ordinal_results.json')
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOptimal k: {optimal_k}")
    print(f"Ordinal: Accuracy={results['mean_accuracy']:.3f}±{results['std_accuracy']:.3f}, MAE={results['mean_mae']:.3f}")
    print(f"RF: Accuracy={rf_clf.cv_results['mean_accuracy']:.3f}±{rf_clf.cv_results['std_accuracy']:.3f}")
    print(f"\nTop {optimal_k} features predicting ripeness:")
    top_k_coef = coef_df.head(optimal_k)
    for i, row in top_k_coef.iterrows():
        direction = "↑ increases" if row['Coefficient'] > 0 else "↓ decreases"
        print(f"  {i+1}. {row['Feature']}: {direction} ripeness (β={row['Coefficient']:.3f})")
    print(f"\nResults: {results_dirs['base']}")
    print("="*70 + "\n")



    
'''
# === ADD THIS RIGHT AFTER STEP 11 (before final summary) ===
    
    # STEP 11.5: Generate extended coefficient plot for visualization
    print("\n" + "="*70)
    print("STEP 11.5: Extended Coefficient Analysis for Visualization")
    print("="*70)
    
    CUSTOM_K_FOR_VIZ = 7  # Change this to 10 or 15 as needed
    
    if CUSTOM_K_FOR_VIZ != optimal_k:
        print(f"\nTraining model with k={CUSTOM_K_FOR_VIZ} features for extended visualization...")
        
        # Select and train with custom k
        X_viz, features_viz, _ = select_k_best_features(X_full, y_raw, k=CUSTOM_K_FOR_VIZ, verbose=False)
        X_viz = pd.DataFrame(X_viz, columns=features_viz)
        
        results_viz = train_evaluate_ordinal(X_viz, y, model_type='LogisticAT', 
                                            scaler='minmax', log_transform=True, 
                                            n_splits=None, verbose=False)
        
        coef_df_viz = results_viz['coefficients']
        
        print(f"Extended model performance (k={CUSTOM_K_FOR_VIZ}):")
        print(f"  Accuracy: {results_viz['mean_accuracy']:.3f}±{results_viz['std_accuracy']:.3f}")
        print(f"  MAE: {results_viz['mean_mae']:.3f}")
        
        print(f"\nExtended coefficients (all {CUSTOM_K_FOR_VIZ} features):")
        for i, row in coef_df_viz.iterrows():
            direction = "↑ increases" if row['Coefficient'] > 0 else "↓ decreases"
            print(f"  {i+1}. {row['Feature']}: {direction} ripeness (β={row['Coefficient']:.3f})")
        
        # Save extended coefficients
        coef_df_viz.to_csv(results_dirs['data'] / f'ordinal_coefficients_k{CUSTOM_K_FOR_VIZ}.csv', index=False)
        
        # Generate extended coefficient plot
        fig, _ = plot_ordinal_coefficients(coef_df_viz, top_n=CUSTOM_K_FOR_VIZ)
        fig.savefig(results_dirs['figures'] / f'ordinal_coefficients_k{CUSTOM_K_FOR_VIZ}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Extended visualization saved: ordinal_coefficients_k{CUSTOM_K_FOR_VIZ}.png")
    else:
        print(f"\nSkipping extended analysis (CUSTOM_K_FOR_VIZ={CUSTOM_K_FOR_VIZ} matches optimal_k={optimal_k})")
    
    # === END SNIPPET ===
'''
if __name__ == "__main__":
    main()