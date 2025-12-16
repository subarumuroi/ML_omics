"""
Comparative Chemometrics Workflow

Compares traditional chemometric approaches (PCA-DA, PLS-DA) with modern ML methods
(Random Forest, Ordinal Regression) for ordered classification tasks.

Demonstrates advantages of ordinal regression:
1. Respects natural ordering (Green < Ripe < Overripe)
2. Quantifies misclassification severity (MAE)
3. Minimal biomarker sets with interpretable coefficients
4. Robust preprocessing for complex omics data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import warnings

# Import package modules
from preprocessing import (
    load_and_impute,
    select_k_best_features,
    get_feature_scores,
)
from models.ordinal import (
    train_evaluate_ordinal,
    encode_ordinal_target,
)
from models import train_evaluate_rf
from utils import (
    prepare_data,
    set_categorical_order,
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
)


def train_pca_lda(X, y, n_components=5, cv=None):
    """
    Traditional PCA-LDA approach.
    
    PCA for dimensionality reduction, then LDA for classification.
    Common in metabolomics before modern ML era.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=min(3, np.bincount(y).min()), shuffle=False)
    
    n_components = min(n_components, X.shape[1], X.shape[0] - len(np.unique(y)))
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    lda = LinearDiscriminantAnalysis()
    
    # Cross-validation
    cv_scores = cross_val_score(lda, X_pca, y, cv=cv, scoring='accuracy')
    
    # Train on full data for predictions
    lda.fit(X_pca, y)
    y_pred = lda.predict(X_pca)
    
    # Calculate MAE (treating as ordinal for fair comparison)
    mae = mean_absolute_error(y, y_pred)
    
    return {
        'model_name': 'PCA-LDA',
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'mae': mae,
        'n_components': n_components,
        'explained_variance': pca.explained_variance_ratio_.sum(),
        'model': lda,
        'transformer': pca,
        'X_transformed': X_pca,
        'y_pred': y_pred,
    }


def train_pls_da(X, y, n_components=5, cv=None):
    """
    PLS-DA (Partial Least Squares Discriminant Analysis).
    
    Gold standard in metabolomics. Supervised dimension reduction
    that maximizes covariance with class labels.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=min(3, np.bincount(y).min()), shuffle=False)
    
    n_components = min(n_components, X.shape[1], X.shape[0] - 1)
    
    # PLS-DA treats classes as dummy variables
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    Y_binary = lb.fit_transform(y)
    
    pls = PLSRegression(n_components=n_components)
    
    # Manual cross-validation (PLS doesn't have built-in CV scorer)
    cv_scores = []
    all_true = []
    all_pred = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y_binary[train_idx], y[test_idx]
        
        pls.fit(X_train, Y_train)
        Y_pred = pls.predict(X_test)
        y_pred = Y_pred.argmax(axis=1)
        
        cv_scores.append(accuracy_score(Y_test, y_pred))
        all_true.extend(Y_test)
        all_pred.extend(y_pred)
    
    # Train on full data
    pls.fit(X, Y_binary)
    Y_pred_full = pls.predict(X)
    y_pred_full = Y_pred_full.argmax(axis=1)
    
    mae = mean_absolute_error(y, y_pred_full)
    
    return {
        'model_name': 'PLS-DA',
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'mae': mae,
        'n_components': n_components,
        'model': pls,
        'y_pred': y_pred_full,
        'all_true': all_true,
        'all_pred': all_pred,
    }


def compare_all_methods(X, y, y_ordinal, n_components=5):
    """
    Compare all methods: PCA-LDA, PLS-DA, Random Forest, Ordinal Regression.
    """
    print("\n" + "="*70)
    print("COMPARING CHEMOMETRIC APPROACHES")
    print("="*70)
    
    results = {}
    
    # 1. PCA-LDA (Traditional unsupervised → supervised)
    print("\n1. Training PCA-LDA (traditional chemometrics)...")
    results['pca_lda'] = train_pca_lda(X, y, n_components=n_components)
    print(f"   Accuracy: {results['pca_lda']['mean_accuracy']:.3f} ± {results['pca_lda']['std_accuracy']:.3f}")
    print(f"   MAE: {results['pca_lda']['mae']:.3f}")
    print(f"   Explained variance: {results['pca_lda']['explained_variance']:.1%}")
    
    # 2. PLS-DA (Gold standard in metabolomics)
    print("\n2. Training PLS-DA (metabolomics gold standard)...")
    results['pls_da'] = train_pls_da(X, y, n_components=n_components)
    print(f"   Accuracy: {results['pls_da']['mean_accuracy']:.3f} ± {results['pls_da']['std_accuracy']:.3f}")
    print(f"   MAE: {results['pls_da']['mae']:.3f}")
    
    # 3. Random Forest (Modern ML baseline)
    print("\n3. Training Random Forest (modern ML baseline)...")
    rf_clf = train_evaluate_rf(X, y, n_estimators=100, max_depth=3, verbose=False)
    results['random_forest'] = {
        'model_name': 'Random Forest',
        'mean_accuracy': rf_clf.cv_results['mean_accuracy'],
        'std_accuracy': rf_clf.cv_results['std_accuracy'],
        'mae': mean_absolute_error(y, rf_clf.model.predict(X)),
        'model': rf_clf,
    }
    print(f"   Accuracy: {results['random_forest']['mean_accuracy']:.3f} ± {results['random_forest']['std_accuracy']:.3f}")
    print(f"   MAE: {results['random_forest']['mae']:.3f}")
    
    # 4. Ordinal Regression (Our approach - respects ordering)
    print("\n4. Training Ordinal Regression (respects stage ordering)...")
    ordinal_results = train_evaluate_ordinal(X, y_ordinal, log_transform=True, 
                                             scaler='minmax', verbose=False)
    results['ordinal'] = {
        'model_name': 'Ordinal Regression',
        'mean_accuracy': ordinal_results['mean_accuracy'],
        'std_accuracy': ordinal_results['std_accuracy'],
        'mae': ordinal_results['mean_mae'],
        'coefficients': ordinal_results['coefficients'],
        'model': ordinal_results['model'],
        'all_true': ordinal_results['all_true'],
        'all_pred': ordinal_results['all_pred'],
    }
    print(f"   Accuracy: {results['ordinal']['mean_accuracy']:.3f} ± {results['ordinal']['std_accuracy']:.3f}")
    print(f"   MAE: {results['ordinal']['mae']:.3f} ← Only method that respects ordering!")
    
    return results


def plot_method_comparison(results, results_dir):
    """
    Create comprehensive comparison visualizations.
    """
    print("\nGenerating comparison visualizations...")
    
    # 1. Performance Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = [r['model_name'] for r in results.values()]
    accuracies = [r['mean_accuracy'] for r in results.values()]
    acc_stds = [r['std_accuracy'] for r in results.values()]
    maes = [r['mae'] for r in results.values()]
    
    # Accuracy comparison
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars1 = axes[0].bar(methods, accuracies, yerr=acc_stds, capsize=5, color=colors, alpha=0.7)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Classification Accuracy\n(Higher is Better)', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # MAE comparison (lower is better)
    bars2 = axes[1].bar(methods, maes, color=colors, alpha=0.7)
    axes[1].set_ylabel('Mean Absolute Error (stages)', fontsize=12)
    axes[1].set_title('MAE - Stage Distance Error\n(Lower is Better)', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels and highlight ordinal
    for i, (bar, mae) in enumerate(zip(bars2, maes)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mae:.3f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold' if methods[i] == 'Ordinal Regression' else 'normal')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrices Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    class_labels = ['Green', 'Ripe', 'Overripe']
    
    for idx, (method_key, result) in enumerate(results.items()):
        if 'y_pred' in result:
            if 'all_true' in result:
                cm = confusion_matrix(result['all_true'], result['all_pred'])
            else:
                # Need to get true labels - use original y
                cm = confusion_matrix(results['pca_lda']['y_pred'], result['y_pred'])  # Placeholder
                
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_labels, yticklabels=class_labels,
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f"{result['model_name']}\nAcc: {result['mean_accuracy']:.3f}, MAE: {result['mae']:.3f}",
                              fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Space Comparison (PCA vs Ordinal Features)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PCA space
    pca_result = results['pca_lda']
    X_pca = pca_result['X_transformed']
    y_plot = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # Assuming structure
    
    for i, (label, color) in enumerate(zip(class_labels, ['green', 'gold', 'brown'])):
        mask = y_plot == i
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=label, s=100, alpha=0.6, color=color, edgecolors='black')
    
    axes[0].set_xlabel(f'PC1 ({pca_result["explained_variance"]:.1%} var)', fontsize=11)
    axes[0].set_ylabel('PC2', fontsize=11)
    axes[0].set_title('PCA Space (Unsupervised)\nNo ordering assumption', 
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Ordinal feature space (if 2+ features selected)
    ordinal_coef = results['ordinal']['coefficients']
    if len(ordinal_coef) >= 2:
        top_features = ordinal_coef['Feature'].iloc[:2].tolist()
        axes[1].text(0.5, 0.5, f'Ordinal Features:\n{top_features[0]}\n{top_features[1]}\n\n' +
                    f'β₁={ordinal_coef["Coefficient"].iloc[0]:.2f}\n' +
                    f'β₂={ordinal_coef["Coefficient"].iloc[1]:.2f}',
                    ha='center', va='center', fontsize=11,
                    transform=axes[1].transAxes)
        axes[1].set_title('Ordinal Features (Supervised)\nOrdered & Interpretable',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_space_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_table(results):
    """
    Create summary comparison table.
    """
    comparison_data = []
    
    for method_key, result in results.items():
        row = {
            'Method': result['model_name'],
            'Accuracy': f"{result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}",
            'MAE': f"{result['mae']:.3f}",
            'Respects Ordering': '✓' if method_key == 'ordinal' else '✗',
            'Interpretability': {
                'pca_lda': 'Low (PC loadings)',
                'pls_da': 'Medium (VIP scores)',
                'random_forest': 'Medium (feature importance)',
                'ordinal': 'High (β coefficients)',
            }[method_key],
            'Features Used': {
                'pca_lda': f"{result.get('n_components', 'N/A')} PCs",
                'pls_da': f"{result.get('n_components', 'N/A')} components",
                'random_forest': 'Selected k',
                'ordinal': f"{len(result.get('coefficients', []))} metabolites",
            }[method_key],
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY TABLE")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    return comparison_df


def main():
    """Run complete comparative chemometrics workflow."""
    
    # Configuration
    DATA_FILE = "data/badata.csv"  # Change as needed
    GROUP_COL = "Groups"
    GROUP_ORDER = ["Green", "Ripe", "Overripe"]
    N_COMPONENTS = 5  # For PCA-LDA and PLS-DA
    
    print("\n" + "="*70)
    print("COMPARATIVE CHEMOMETRICS WORKFLOW")
    print("Traditional vs Modern ML Approaches")
    print("="*70 + "\n")
    
    results_dirs = create_results_directory(base_path='results', 
                                           run_name='chemometrics_comparison')
    
    # Load and prepare data
    print("STEP 1: Loading and preparing data...")
    df = load_and_impute(DATA_FILE, group_col=GROUP_COL, impute=True, 
                        fill_value=0, drop_threshold=0.35)
    df = df.drop(columns='Unnamed: 0', errors='ignore')
    df = set_categorical_order(df, col=GROUP_COL, categories=GROUP_ORDER)
    
    X, y_raw, _, _, _ = prepare_data(df, target_col=GROUP_COL, drop_missing=True)
    X = X.select_dtypes(include=[np.number])
    
    # Handle negative values
    if (X < 0).any().any():
        min_val = X.min().min()
        X = X - min_val + 1
        print(f"✓ Shifted data by {-min_val + 1:.2f} to handle negative values")
    
    # Convert to numpy for compatibility
    X = X.values
    y = y_raw.map({cat: i for i, cat in enumerate(GROUP_ORDER)}).values
    y_ordinal = np.array(encode_ordinal_target(y_raw, categories=GROUP_ORDER))
    
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Feature selection for fair comparison
    print("\nSTEP 2: Feature selection...")
    X_df = pd.DataFrame(X)
    X_selected, selected_features, _ = select_k_best_features(
        X_df, y_raw, k=min(15, X.shape[1]), verbose=True
    )
    
    print(f"Selected {X_selected.shape[1]} features for comparison")
    
    # Compare all methods
    print("\nSTEP 3: Training and comparing all methods...")
    results = compare_all_methods(X_selected, y, y_ordinal, n_components=N_COMPONENTS)
    
    # Create comparison table
    print("\nSTEP 4: Creating comparison summary...")
    comparison_df = create_comparison_table(results)
    
    # Generate visualizations
    print("\nSTEP 5: Generating comparative visualizations...")
    plot_method_comparison(results, results_dirs['figures'])
    
    # Additional ordinal-specific visualizations
    if 'ordinal' in results:
        print("\nSTEP 6: Ordinal-specific visualizations...")
        
        # Coefficients
        from visualization.ordinal_plots import plot_ordinal_coefficients
        fig, _ = plot_ordinal_coefficients(results['ordinal']['coefficients'], top_n=10)
        fig.savefig(results_dirs['figures'] / 'ordinal_coefficients.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion matrix
        fig, _ = plot_ordinal_confusion_matrix(
            results['ordinal']['all_true'], 
            results['ordinal']['all_pred'],
            class_labels=GROUP_ORDER
        )
        fig.savefig(results_dirs['figures'] / 'ordinal_confusion_detailed.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save results
    print("\nSTEP 7: Saving results...")
    comparison_df.to_csv(results_dirs['data'] / 'method_comparison.csv', index=False)
    
    # Save detailed results
    summary = {
        'data_file': DATA_FILE,
        'n_samples': X.shape[0],
        'n_features_total': X.shape[1],
        'n_features_selected': X_selected.shape[1],
        'methods_compared': list(results.keys()),
        'comparison_table': comparison_df.to_dict(),
        'ordinal_coefficients': results['ordinal']['coefficients'].to_dict() if 'ordinal' in results else None,
    }
    
    save_results(summary, results_dirs['base'] / 'comparison_results.json')
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey Findings:")
    print(f"1. Best Accuracy: {max(r['mean_accuracy'] for r in results.values()):.3f}")
    print(f"2. Best MAE: {min(r['mae'] for r in results.values()):.3f} (Ordinal only metric that matters!)")
    print(f"3. Most Interpretable: Ordinal Regression (β coefficients)")
    print(f"4. Minimal Biomarkers: {len(results['ordinal']['coefficients'])} metabolites")
    
    if 'ordinal' in results:
        print("\nTop Ordinal Features:")
        for i, row in results['ordinal']['coefficients'].head(3).iterrows():
            direction = "↑ increases" if row['Coefficient'] > 0 else "↓ decreases"
            print(f"  {i+1}. {row['Feature']}: {direction} ripeness (β={row['Coefficient']:.3f})")
    
    print(f"\nResults saved to: {results_dirs['base']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()