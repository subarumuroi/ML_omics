"""
Complete analysis workflow demonstrating the full package.

This example shows:
1. Data loading and preprocessing
2. Feature selection
3. Model training with cross-validation
4. Permutation testing
5. Feature importance analysis
6. Comprehensive visualizations
7. Results saving
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Add src to path (adjust if needed)
sys.path.insert(0, '../src')

# Import package modules
from preprocessing.data_processing import load_and_impute
from preprocessing.feature_selection import (
    select_k_best_features,
    get_top_n_features,
)
from models.random_forest import train_evaluate_model
from models.evaluation import create_results_summary, compare_models
from analysis.importance import (
    get_feature_importance_df,
    compute_permutation_importance,
    get_shap_values,
)
from analysis.statistical_tests import (
    permutation_test,
    permutation_test_with_feature_subset,
    compare_permutation_tests,
)
from visualization.model_plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance_boxplot,
    plot_permutation_test_distribution,
    plot_cv_scores,
)
from visualization.data_plots import (
    plot_compound_trends,
    plot_compound_boxplots,
    plot_compound_heatmap,
)
from visualization.dimension_reduction import (
    plot_tsne,
    plot_umap,
    plot_shap_summary,
)
from utils.data_utils import (
    prepare_data,
    set_categorical_order,
    print_data_summary,
    check_class_balance,
    drop_index_columns,
)
from utils.io_utils import (
    save_results,
    save_dataframe,
    create_results_directory,
)


def main():
    """Run complete analysis workflow."""
    
    # ======================================================================
    # CONFIGURATION
    # ======================================================================
    DATA_FILE = "badata_imputed.csv"
    GROUP_COL = "Groups"
    GROUP_ORDER = ["Green", "Ripe", "Overripe"]
    N_FEATURES_SELECT = 15
    N_TOP_FEATURES = 3  # For detailed analysis
    N_PERMUTATIONS = 1000
    RANDOM_STATE = 42
    
    # Create results directory
    results_dirs = create_results_directory(base_path='results', run_name='banana_analysis')
    
    # ======================================================================
    # 1. LOAD AND PREPROCESS DATA
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    # Load data with imputation
    df = load_and_impute(DATA_FILE, group_col=GROUP_COL, impute=True, fill_value=0)
    
    # Drop index columns
    df = drop_index_columns(df)
    
    # Set categorical order
    df = set_categorical_order(df, col=GROUP_COL, categories=GROUP_ORDER)
    
    # Prepare data
    X, y_raw, y, le, dropped_cols = prepare_data(
        df, 
        target_col=GROUP_COL, 
        drop_missing=True, 
        verbose=True
    )
    
    # Print summary
    print_data_summary(X, y, le)
    check_class_balance(y, le)
    
    # ======================================================================
    # 2. FEATURE SELECTION
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 2: FEATURE SELECTION")
    print("="*70)
    
    X_selected, selected_features, selector = select_k_best_features(
        X, y, k=N_FEATURES_SELECT, verbose=True
    )
    X_df = pd.DataFrame(X_selected, columns=selected_features)
    
    # Save selected features
    save_dataframe(
        pd.DataFrame({'Feature': selected_features}),
        results_dirs['data'] / 'selected_features.csv'
    )
    
    # ======================================================================
    # 3. TRAIN MODEL WITH CROSS-VALIDATION
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING (FULL FEATURE SET)")
    print("="*70)
    
    clf_full = train_evaluate_model(
        X=X_df,
        y=y,
        n_estimators=100,
        max_depth=3,
        n_splits=None,  # Auto-detect
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    # Plot CV scores
    fig_cv, _ = plot_cv_scores(
        clf_full.cv_results['scores'],
        clf_full.cv_results['balanced_scores']
    )
    fig_cv.savefig(results_dirs['figures'] / 'cv_scores.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cv)
    
    # Plot confusion matrix
    fig_cm, _ = plot_confusion_matrix(
        clf_full.confusion_matrix,
        class_labels=le.classes_,
        title="Confusion Matrix (CV Aggregated)"
    )
    fig_cm.savefig(results_dirs['figures'] / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    
    # ======================================================================
    # 4. FEATURE IMPORTANCE ANALYSIS
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Tree-based importance
    importance_df = get_feature_importance_df(
        clf_full.model,
        selected_features,
        sort=True
    )
    
    print("\nTop 10 features by tree-based importance:")
    print(importance_df.head(10))
    
    # Save importance
    save_dataframe(
        importance_df,
        results_dirs['data'] / 'feature_importance.csv'
    )
    
    # Plot importance
    fig_imp, _ = plot_feature_importance(
        importance_df,
        top_n=10,
        title="Top 10 Features by Tree-Based Importance"
    )
    fig_imp.savefig(results_dirs['figures'] / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig_imp)
    
    # Permutation importance
    print("\nComputing permutation importance...")
    perm_df, perm_result = compute_permutation_importance(
        clf_full.model,
        X_df,
        y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    # Plot permutation importance
    fig_perm, _ = plot_permutation_importance_boxplot(
        perm_result,
        selected_features,
        top_n=10
    )
    fig_perm.savefig(results_dirs['figures'] / 'permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig_perm)
    
    # ======================================================================
    # 5. PERMUTATION TEST (FULL MODEL)
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 5: PERMUTATION TEST (FULL MODEL)")
    print("="*70)
    
    n_splits = np.bincount(y).min()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
    
    perm_results_full = permutation_test(
        model=rf,
        X=X_df,
        y=y,
        cv=cv,
        n_permutations=N_PERMUTATIONS,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    # Plot permutation test
    fig_perm_test, _ = plot_permutation_test_distribution(
        perm_results_full['perm_accuracies'],
        perm_results_full['observed_accuracy']
    )
    fig_perm_test.savefig(results_dirs['figures'] / 'permutation_test_full.png', dpi=300, bbox_inches='tight')
    plt.close(fig_perm_test)
    
    # ======================================================================
    # 6. COMPOUND TREND ANALYSIS
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 6: COMPOUND TREND ANALYSIS")
    print("="*70)
    
    top_features = get_top_n_features(importance_df, n=10)
    
    # Linear scale
    fig_linear, _ = plot_compound_trends(
        df,
        top_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER,
        scale='linear'
    )
    fig_linear.savefig(results_dirs['figures'] / 'compound_trends_linear.png', dpi=300, bbox_inches='tight')
    plt.close(fig_linear)
    
    # Log scale
    fig_log, _ = plot_compound_trends(
        df,
        top_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER,
        scale='log'
    )
    fig_log.savefig(results_dirs['figures'] / 'compound_trends_log.png', dpi=300, bbox_inches='tight')
    plt.close(fig_log)
    
    # Heatmap
    fig_heat, _ = plot_compound_heatmap(
        df,
        top_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER
    )
    fig_heat.savefig(results_dirs['figures'] / 'compound_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig_heat)
    
    # ======================================================================
    # 7. TOP N FEATURES ANALYSIS
    # ======================================================================
    print(f"\n" + "="*70)
    print(f"STEP 7: ANALYSIS WITH TOP {N_TOP_FEATURES} FEATURES")
    print("="*70)
    
    top_n_features = get_top_n_features(importance_df, n=N_TOP_FEATURES)
    print(f"\nTop {N_TOP_FEATURES} features:")
    for i, feat in enumerate(top_n_features, 1):
        print(f"  {i}. {feat}")
    
    # Train model with top N
    X_topn = X_df[top_n_features]
    clf_topn = train_evaluate_model(
        X_topn, y, 
        n_estimators=100, 
        max_depth=3,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    # Plot trends for top N
    fig_topn_linear, _ = plot_compound_trends(
        df,
        top_n_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER,
        scale='linear'
    )
    plt.title(f"Top {N_TOP_FEATURES} Compounds (Linear Scale)")
    fig_topn_linear.savefig(results_dirs['figures'] / f'top{N_TOP_FEATURES}_trends_linear.png', dpi=300, bbox_inches='tight')
    plt.close(fig_topn_linear)
    
    # Boxplots
    fig_box, _ = plot_compound_boxplots(
        df,
        top_n_features,
        group_col=GROUP_COL,
        group_order=GROUP_ORDER
    )
    plt.title(f"Top {N_TOP_FEATURES} Compounds Distribution")
    fig_box.savefig(results_dirs['figures'] / f'top{N_TOP_FEATURES}_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close(fig_box)
    
    # Permutation test for top N
    perm_results_topn = permutation_test_with_feature_subset(
        model=rf,
        X_full=X_df,
        feature_names=top_n_features,
        y=y,
        n_splits=n_splits,
        n_permutations=N_PERMUTATIONS,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    fig_perm_topn, _ = plot_permutation_test_distribution(
        perm_results_topn['perm_accuracies'],
        perm_results_topn['observed_accuracy']
    )
    plt.title(f"Permutation Test (Top {N_TOP_FEATURES} Features)")
    fig_perm_topn.savefig(results_dirs['figures'] / f'permutation_test_top{N_TOP_FEATURES}.png', dpi=300, bbox_inches='tight')
    plt.close(fig_perm_topn)
    
    # ======================================================================
    # 8. SHAP ANALYSIS
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 8: SHAP ANALYSIS")
    print("="*70)
    
    try:
        explainer, shap_values = get_shap_values(clf_full.model, X_df)
        plot_shap_summary(
            shap_values, 
            X_df, 
            class_labels=le.classes_,
            save_path=str(results_dirs['figures'] / 'shap_summary.png')
        )
    except ImportError:
        print("SHAP not installed. Skipping SHAP analysis.")
    
    # ======================================================================
    # 9. DIMENSIONALITY REDUCTION
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 9: DIMENSIONALITY REDUCTION")
    print("="*70)
    
    # t-SNE
    print("Computing t-SNE...")
    fig_tsne, _, _ = plot_tsne(
        X_df, y, 
        label_encoder=le, 
        perplexity=5,
        random_state=RANDOM_STATE
    )
    fig_tsne.savefig(results_dirs['figures'] / 'tsne.png', dpi=300, bbox_inches='tight')
    plt.close(fig_tsne)
    
    # UMAP
    try:
        print("Computing UMAP...")
        fig_umap, _, _ = plot_umap(
            X_df, y, 
            label_encoder=le, 
            n_neighbors=10,
            random_state=RANDOM_STATE
        )
        fig_umap.savefig(results_dirs['figures'] / 'umap.png', dpi=300, bbox_inches='tight')
        plt.close(fig_umap)
    except ImportError:
        print("UMAP not installed. Skipping UMAP analysis.")
    
    # ======================================================================
    # 10. SAVE RESULTS
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 10: SAVING RESULTS")
    print("="*70)
    
    # Create summaries
    summary_full = create_results_summary(
        clf_full.cv_results, 
        perm_results_full,
        {'n_features': len(selected_features)}
    )
    
    summary_topn = create_results_summary(
        clf_topn.cv_results,
        perm_results_topn,
        {'n_features': N_TOP_FEATURES, 'features': top_n_features}
    )
    
    # Compare models
    comparison = compare_models(
        [summary_full, summary_topn],
        model_names=[f'Full Model ({N_FEATURES_SELECT} features)', 
                    f'Top {N_TOP_FEATURES} Features']
    )
    
    print("\nModel Comparison:")
    print(comparison)
    
    # Save all results
    all_results = {
        'full_model': summary_full,
        f'top_{N_TOP_FEATURES}_model': summary_topn,
        'comparison': comparison.to_dict(),
        'selected_features': list(selected_features),
        'top_features': top_n_features,
        'config': {
            'n_features_select': N_FEATURES_SELECT,
            'n_top_features': N_TOP_FEATURES,
            'n_permutations': N_PERMUTATIONS,
            'random_state': RANDOM_STATE,
        }
    }
    
    save_results(all_results, results_dirs['base'] / 'analysis_results.json')
    save_dataframe(comparison, results_dirs['data'] / 'model_comparison.csv')
    
    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    print(f"\nFull Model ({N_FEATURES_SELECT} features):")
    print(f"  CV Accuracy: {summary_full['cv_accuracy']:.3f} ± {summary_full['cv_accuracy_std']:.3f}")
    print(f"  Permutation p-value: {summary_full['permutation_p_value']:.4f}")
    
    print(f"\nTop {N_TOP_FEATURES} Features Model:")
    print(f"  Features: {', '.join(top_n_features)}")
    print(f"  CV Accuracy: {summary_topn['cv_accuracy']:.3f} ± {summary_topn['cv_accuracy_std']:.3f}")
    print(f"  Permutation p-value: {summary_topn['permutation_p_value']:.4f}")
    
    print(f"\nResults saved to: {results_dirs['base']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()