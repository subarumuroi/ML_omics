"""
Random forest workflow using the installed package.

After installing with: pip install -e .

This example demonstrates clean imports without sys.path hacks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Clean imports using the installed package
from preprocessing import load_and_impute, select_k_best_features, get_top_n_features
from models import train_evaluate_model
from models.evaluation import create_results_summary, compare_models
from analysis import (
    get_feature_importance_df,
    compute_permutation_importance,
    permutation_test,
    permutation_test_with_feature_subset,
    get_shap_values,
)
from visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_test_distribution,
    plot_compound_trends,
    plot_compound_boxplots,
    plot_tsne,
    plot_umap,
    plot_shap_summary,
)
from utils import (
    prepare_data,
    set_categorical_order,
    print_data_summary,
    create_results_directory,
    save_results,
    save_dataframe,
)


def main():
    """Run complete analysis workflow."""
    
    # ======================================================================
    # CONFIGURATION
    # ======================================================================
    DATA_FILE = "data\\badata.csv"
    GROUP_COL = "Groups"
    GROUP_ORDER = ["Green", "Ripe", "Overripe"]
    N_FEATURES_SELECT = 15
    N_TOP_FEATURES = 3
    N_PERMUTATIONS = 1000
    RANDOM_STATE = 42
    
    print("\n" + "="*70)
    print("ML ANALYSIS WORKFLOW")
    print("Using installed package with modern imports")
    print("="*70 + "\n")
    
    # Create results directory
    results_dirs = create_results_directory(
        base_path='results', 
        run_name='banana_analysis'
    )
    
    # ======================================================================
    # 1. LOAD AND PREPARE DATA
    # ======================================================================
    print("STEP 1: Loading and preparing data...")
    
    df = load_and_impute(DATA_FILE, group_col=GROUP_COL, impute=True, fill_value=0)
    df = set_categorical_order(df, col=GROUP_COL, categories=GROUP_ORDER)
    X, y_raw, y, le, _ = prepare_data(df, target_col=GROUP_COL)
    
    print_data_summary(X, y, le)
    
    # ======================================================================
    # 2. FEATURE SELECTION
    # ======================================================================
    print("\nSTEP 2: Selecting features...")
    
    X_selected, selected_features, _ = select_k_best_features(X, y, k=N_FEATURES_SELECT)
    X_df = pd.DataFrame(X_selected, columns=selected_features)
    
    # ======================================================================
    # 3. TRAIN MODEL
    # ======================================================================
    print("\nSTEP 3: Training model with cross-validation...")
    
    clf = train_evaluate_model(
        X=X_df,
        y=y,
        n_estimators=100,
        max_depth=3,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    # Visualize
    fig, _ = plot_confusion_matrix(clf.confusion_matrix, le.classes_)
    fig.savefig(results_dirs['figures'] / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 4. FEATURE IMPORTANCE
    # ======================================================================
    print("\nSTEP 4: Analyzing feature importance...")
    
    importance_df = get_feature_importance_df(clf.model, selected_features)
    
    fig, _ = plot_feature_importance(importance_df, top_n=10)
    fig.savefig(results_dirs['figures'] / 'importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 5. PERMUTATION TEST
    # ======================================================================
    print("\nSTEP 5: Running permutation test...")
    
    cv = StratifiedKFold(n_splits=np.bincount(y).min(), shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
    
    perm_results = permutation_test(
        model=rf,
        X=X_df,
        y=y,
        cv=cv,
        n_permutations=N_PERMUTATIONS,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    fig, _ = plot_permutation_test_distribution(
        perm_results['perm_accuracies'],
        perm_results['observed_accuracy']
    )
    fig.savefig(results_dirs['figures'] / 'permutation_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 6. COMPOUND TRENDS
    # ======================================================================
    print("\nSTEP 6: Visualizing compound trends...")
    
    top_features = get_top_n_features(importance_df, n=10)
    
    fig, _ = plot_compound_trends(df, top_features, scale='log')
    fig.savefig(results_dirs['figures'] / 'trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 7. TOP N FEATURES ANALYSIS
    # ======================================================================
    print(f"\nSTEP 7: Analyzing top {N_TOP_FEATURES} features...")
    
    top_n = get_top_n_features(importance_df, n=N_TOP_FEATURES)
    print(f"Top {N_TOP_FEATURES}: {', '.join(top_n)}")
    
    X_topn = X_df[top_n]
    clf_topn = train_evaluate_model(X_topn, y, random_state=RANDOM_STATE)
    
    perm_results_topn = permutation_test_with_feature_subset(
        rf, X_df, top_n, y, n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE
    )
    
    fig, _ = plot_compound_boxplots(df, top_n)
    fig.savefig(results_dirs['figures'] / f'top{N_TOP_FEATURES}_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # 8. DIMENSIONALITY REDUCTION
    # ======================================================================
    print("\nSTEP 8: Dimensionality reduction...")
    
    fig_tsne, _, _ = plot_tsne(X_df, y, label_encoder=le, random_state=RANDOM_STATE)
    fig_tsne.savefig(results_dirs['figures'] / 'tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        fig_umap, _, _ = plot_umap(X_df, y, label_encoder=le, random_state=RANDOM_STATE)
        fig_umap.savefig(results_dirs['figures'] / 'umap.png', dpi=300, bbox_inches='tight')
        plt.close()
    except ImportError:
        print("UMAP not installed, skipping...")
    
    # ======================================================================
    # 9. SHAP ANALYSIS
    # ======================================================================
    print("\nSTEP 9: SHAP analysis...")
    
    try:
        explainer, shap_values = get_shap_values(clf.model, X_df)
        plot_shap_summary(shap_values, X_df, le.classes_)
    except ImportError:
        print("SHAP not installed, skipping...")
    
    # ======================================================================
    # 10. SAVE RESULTS
    # ======================================================================
    print("\nSTEP 10: Saving results...")
    
    summary_full = create_results_summary(clf.cv_results, perm_results)
    summary_topn = create_results_summary(clf_topn.cv_results, perm_results_topn)
    
    comparison = compare_models(
        [summary_full, summary_topn],
        [f'Full ({N_FEATURES_SELECT} features)', f'Top {N_TOP_FEATURES}']
    )
    
    print("\nModel Comparison:")
    print(comparison)
    
    results = {
        'full_model': summary_full,
        f'top_{N_TOP_FEATURES}_model': summary_topn,
        'comparison': comparison.to_dict(),
        'selected_features': list(selected_features),
        'top_features': top_n,
    }
    
    save_results(results, results_dirs['base'] / 'results.json')
    save_dataframe(comparison, results_dirs['data'] / 'comparison.csv')
    
    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nFull Model: Accuracy = {summary_full['cv_accuracy']:.3f}, p = {summary_full['permutation_p_value']:.4f}")
    print(f"Top {N_TOP_FEATURES}: Accuracy = {summary_topn['cv_accuracy']:.3f}, p = {summary_topn['permutation_p_value']:.4f}")
    print(f"\nResults saved to: {results_dirs['base']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()