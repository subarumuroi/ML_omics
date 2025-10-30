"""
Example workflow demonstrating complete analysis pipeline.

This script shows how to use all the package modules together to:
1. Load and prepare data
2. Select features
3. Train model with cross-validation
4. Perform permutation testing
5. Generate visualizations
6. Analyze feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import shap

# Import package modules
from models import RFClassifierCV, train_evaluate_model
from feature_selection import (
    select_k_best_features,
    get_feature_importance_df,
    compute_permutation_importance,
    get_top_n_features,
)
from statistical_tests import (
    permutation_test,
    permutation_test_with_feature_subset,
)
from visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance_boxplot,
    plot_compound_trends,
    plot_compound_boxplots,
    plot_permutation_test_distribution,
    plot_tsne,
    plot_umap,
    plot_shap_summary,
)
from utils import (
    prepare_data,
    set_categorical_order,
    print_data_summary,
    create_results_summary,
)


def main():
    """Run complete analysis workflow."""
    
    # ======================================================================
    # 1. LOAD AND PREPARE DATA
    # ======================================================================
    print("Loading data...")
    df = pd.read_csv("badata_imputed.csv")
    
    # Drop index column if present
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(columns=[df.columns[0]])
    
    # Set categorical order for groups
    df = set_categorical_order(
        df, 
        col='Groups',
        categories=['Green', 'Ripe', 'Overripe'],
        ordered=True
    )
    
    # Prepare data
    X, y_raw, y, le, dropped_cols = prepare_data(df, target_col='Groups')
    
    # Print summary
    print_data_summary(X, y, le)
    
    # ======================================================================
    # 2. FEATURE SELECTION
    # ======================================================================
    print("Selecting features...")
    k = 15  # Number of features to select
    X_selected, selected_features, selector = select_k_best_features(X, y, k=k)
    X_df = pd.DataFrame(X_selected, columns=selected_features)
    
    # ======================================================================
    # 3. TRAIN MODEL WITH CROSS-VALIDATION
    # ======================================================================
    print("\n" + "="*60)
    print("TRAINING MODEL WITH CROSS-VALIDATION")
    print("="*60)
    
    clf = train_evaluate_model(
        X=X_df,
        y=y,
        n_estimators=100,
        max_depth=3,
        n_splits=None,  # Auto-detect
        random_state=42,
        verbose=True
    )
    
    # ======================================================================
    # 4. VISUALIZE RESULTS
    # ======================================================================
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        clf.confusion_matrix,
        class_labels=le.classes_,
        title="Confusion Matrix (CV Aggregated)"
    )
    plt.show()
    
    # Feature importance
    importance_df = get_feature_importance_df(
        clf.model,
        selected_features,
        sort=True
    )
    
    plot_feature_importance(
        importance_df,
        top_n=10,
        title="Top 10 Features by Tree-Based Importance"
    )
    plt.show()
    
    # ======================================================================
    # 5. PERMUTATION IMPORTANCE
    # ======================================================================
    print("\nComputing permutation importance...")
    perm_df, perm_result = compute_permutation_importance(
        clf.model,
        X_df,
        y,
        n_repeats=10,
        random_state=42
    )
    
    plot_permutation_importance_boxplot(
        perm_result,
        selected_features,
        top_n=10
    )
    plt.show()
    
    # ======================================================================
    # 6. COMPOUND TRENDS
    # ======================================================================
    print("\nPlotting compound trends...")
    top_features = get_top_n_features(importance_df, n=10)
    
    # Linear scale
    plot_compound_trends(
        df,
        top_features,
        group_col='Groups',
        group_order=['Green', 'Ripe', 'Overripe'],
        scale='linear'
    )
    plt.show()
    
    # Log scale
    plot_compound_trends(
        df,
        top_features,
        group_col='Groups',
        group_order=['Green', 'Ripe', 'Overripe'],
        scale='log'
    )
    plt.show()
    
    # ======================================================================
    # 7. PERMUTATION TEST (FULL MODEL)
    # ======================================================================
    print("\n" + "="*60)
    print("PERMUTATION TEST - FULL MODEL")
    print("="*60)
    
    n_splits = np.bincount(y).min()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    perm_results = permutation_test(
        model=rf,
        X=X_df,
        y=y,
        cv=cv,
        n_permutations=1000,
        random_state=42,
        verbose=True
    )
    
    plot_permutation_test_distribution(
        perm_results['perm_accuracies'],
        perm_results['observed_accuracy']
    )
    plt.show()
    
    # ======================================================================
    # 8. ANALYSIS WITH TOP N FEATURES
    # ======================================================================
    n_top = 3  # Can be changed to any number
    print(f"\n" + "="*60)
    print(f"ANALYSIS WITH TOP {n_top} FEATURES")
    print("="*60)
    
    top_n_features = get_top_n_features(importance_df, n=n_top)
    print(f"\nTop {n_top} features:")
    for i, feat in enumerate(top_n_features, 1):
        print(f"  {i}. {feat}")
    
    # Train model with top N features
    X_topn = X_df[top_n_features]
    
    clf_topn = RFClassifierCV(n_estimators=100, max_depth=3, random_state=42)
    cv_results_topn = clf_topn.cross_validate(X_topn, y, verbose=True)
    clf_topn.fit(X_topn, y)
    
    # Plot trends for top N
    plot_compound_trends(
        df,
        top_n_features,
        group_col='Groups',
        group_order=['Green', 'Ripe', 'Overripe'],
        scale='linear'
    )
    plt.title(f"Top {n_top} Compounds Across Ripeness (Linear Scale)")
    plt.show()
    
    # Boxplots
    plot_compound_boxplots(
        df,
        top_n_features,
        group_col='Groups',
        group_order=['Green', 'Ripe', 'Overripe']
    )
    plt.title(f"Top {n_top} Compounds Distribution by Ripeness")
    plt.show()
    
    # Permutation test for top N features
    print(f"\nPermutation test with top {n_top} features...")
    perm_results_topn = permutation_test_with_feature_subset(
        model=rf,
        X_full=X_df,
        feature_names=top_n_features,
        y=y,
        n_splits=n_splits,
        n_permutations=1000,
        random_state=42,
        verbose=True
    )
    
    plot_permutation_test_distribution(
        perm_results_topn['perm_accuracies'],
        perm_results_topn['observed_accuracy']
    )
    plt.title(f"Permutation Test (Top {n_top} Features)")
    plt.show()
    
    # ======================================================================
    # 9. SHAP ANALYSIS
    # ======================================================================
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)
    
    explainer = shap.TreeExplainer(clf.model)
    shap_values = explainer.shap_values(X_df)
    
    plot_shap_summary(shap_values, X_df, class_labels=le.classes_)
    
    # ======================================================================
    # 10. DIMENSIONALITY REDUCTION
    # ======================================================================
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION")
    print("="*60)
    
    # t-SNE
    print("Computing t-SNE...")
    plot_tsne(X_df, y, label_encoder=le, perplexity=5)
    plt.show()
    
    # UMAP (if available)
    try:
        print("Computing UMAP...")
        plot_umap(X_df, y, label_encoder=le, n_neighbors=10)
        plt.show()
    except ImportError:
        print("UMAP not available. Install with: pip install umap-learn")
    
    # ======================================================================
    # 11. SUMMARY
    # ======================================================================
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    summary = create_results_summary(cv_results_topn, perm_results_topn)
    
    print(f"\nFull Model ({len(selected_features)} features):")
    print(f"  CV Accuracy: {clf.cv_results['mean_accuracy']:.3f} ± {clf.cv_results['std_accuracy']:.3f}")
    print(f"  Permutation p-value: {perm_results['p_value']:.4f}")
    
    print(f"\nTop {n_top} Features Model:")
    print(f"  CV Accuracy: {summary['cv_accuracy']:.3f} ± {summary['cv_accuracy_std']:.3f}")
    print(f"  Permutation p-value: {summary['permutation_p_value']:.4f}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()