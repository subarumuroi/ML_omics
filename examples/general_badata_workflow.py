"""
General Classification Workflow for Metabolomics Data (Badata).

This version implements proper nested cross-validation to prevent data leakage.
Feature selection happens INSIDE each CV fold, not before.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Import package modules
from preprocessing import (
    load_and_impute,
)
from models.ordinal import (
    encode_ordinal_target,
)
from models import train_evaluate_rf 
from models.pls_da import train_evaluate_plsda
from models.svm import train_evaluate_svm
from models.ordinal import train_evaluate_ordinal

from utils import (
    prepare_data,
    set_categorical_order,
    create_results_directory,
    save_results,
)
from visualization.ordinal_plots import (
    plot_ordinal_model_comparison,
)


def select_features_on_fold(X_train, y_train, k=3):
    """
    Select top k features using ONLY training data.
    Returns selected feature indices.
    """
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    return selector.get_support(indices=True)


def nested_cv_with_feature_selection(X, y, y_raw, k_features, model_type='ordinal', verbose=True):
    """
    Perform proper nested CV where feature selection happens inside each fold.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Full feature matrix
    y : array-like, shape (n_samples,)
        Encoded target labels
    y_raw : array-like, shape (n_samples,)
        Raw target labels (for PLS-DA)
    k_features : int
        Number of features to select
    model_type : str
        'ordinal', 'rf', 'plsda', or 'svm'
    
    Returns:
    --------
    dict with accuracy scores, MAE scores, and selected features per fold
    """
    loo = LeaveOneOut()
    
    accuracies = []
    maes = []
    selected_features_per_fold = []
    
    fold_num = 0
    for train_idx, test_idx in loo.split(X):
        fold_num += 1
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Feature selection on training data ONLY
        feature_indices = select_features_on_fold(X_train.values, y_train, k=k_features)
        selected_features_per_fold.append(X_train.columns[feature_indices].tolist())
        
        # Apply feature selection
        X_train_selected = X_train.iloc[:, feature_indices]
        X_test_selected = X_test.iloc[:, feature_indices]
        
        # Handle negative values for log transformation
        if model_type == 'ordinal':
            for col_idx in range(X_train_selected.shape[1]):
                col_data = X_train_selected.iloc[:, col_idx]
                if (col_data < 0).any():
                    shift = abs(col_data.min()) + 1
                    X_train_selected.iloc[:, col_idx] = col_data + shift
                    X_test_selected.iloc[:, col_idx] = X_test_selected.iloc[:, col_idx] + shift
        
        # Scale data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train and evaluate model
        if model_type == 'ordinal':
            from mord import LogisticAT
            model = LogisticAT()
            model.fit(np.log1p(X_train_scaled), y_train)
            y_pred = model.predict(np.log1p(X_test_scaled))
            
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
        elif model_type == 'plsda':
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.linear_model import LogisticRegression
            # Use raw labels for PLS-DA
            y_train_raw = y_raw.iloc[train_idx]
            y_test_raw = y_raw.iloc[test_idx]
            
            pls = PLSRegression(n_components=min(2, X_train_scaled.shape[1], X_train_scaled.shape[0]-1))
            pls.fit(X_train_scaled, pd.get_dummies(y_train_raw))
            X_train_pls = pls.transform(X_train_scaled)
            X_test_pls = pls.transform(X_test_scaled)
            
            lr = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
            lr.fit(X_train_pls, y_train)
            y_pred = lr.predict(X_test_pls)
            
        elif model_type == 'svm':
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', C=1, gamma='scale')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        acc = (y_pred == y_test).mean()
        mae = np.abs(y_pred - y_test).mean()
        
        accuracies.append(acc)
        maes.append(mae)
        
        if verbose:
            print(f"Fold {fold_num}: Accuracy={acc:.3f}, MAE={mae:.3f}, Features={selected_features_per_fold[-1]}")
    
    results = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_mae': np.mean(maes),
        'std_mae': np.std(maes),
        'accuracies': accuracies,
        'maes': maes,
        'selected_features_per_fold': selected_features_per_fold
    }
    
    if verbose:
        print(f"\nMean CV Accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        print(f"Mean CV MAE: {results['mean_mae']:.3f} ± {results['std_mae']:.3f}")
    
    return results


def run_nested_model_comparison(X, y_encoded, y_raw, k_features, results_dirs, verbose=True):
    """
    Run all four models with proper nested CV and feature selection.
    """
    all_results = {}
    
    # 1. Ordinal Regression
    if verbose: 
        print("\n" + "="*70)
        print("1/4: Running Ordinal Regression (LogisticAT) with Nested CV")
        print("="*70)
    results_ord = nested_cv_with_feature_selection(
        X, y_encoded, y_raw, k_features, model_type='ordinal', verbose=verbose
    )
    all_results['Ordinal_LogisticAT'] = {
        'Model': 'Ordinal (LogAT)',
        'Mean_Accuracy': results_ord['mean_accuracy'],
        'Std_Accuracy': results_ord['std_accuracy'],
        'MAE': results_ord['mean_mae'],
        'Std_MAE': results_ord['std_mae'],
        'Selected_Features': results_ord['selected_features_per_fold']
    }
    
    # 2. Random Forest
    if verbose: 
        print("\n" + "="*70)
        print("2/4: Running Random Forest with Nested CV")
        print("="*70)
    results_rf = nested_cv_with_feature_selection(
        X, y_encoded, y_raw, k_features, model_type='rf', verbose=verbose
    )
    all_results['Random_Forest'] = {
        'Model': 'Random Forest',
        'Mean_Accuracy': results_rf['mean_accuracy'],
        'Std_Accuracy': results_rf['std_accuracy'],
        'MAE': results_rf['mean_mae'],
        'Std_MAE': results_rf['std_mae'],
        'Selected_Features': results_rf['selected_features_per_fold']
    }
    
    # 3. PLS-DA
    if verbose: 
        print("\n" + "="*70)
        print("3/4: Running PLS-DA with Nested CV")
        print("="*70)
    results_plsda = nested_cv_with_feature_selection(
        X, y_encoded, y_raw, k_features, model_type='plsda', verbose=verbose
    )
    all_results['PLS_DA'] = {
        'Model': 'PLS-DA',
        'Mean_Accuracy': results_plsda['mean_accuracy'],
        'Std_Accuracy': results_plsda['std_accuracy'],
        'MAE': results_plsda['mean_mae'],
        'Std_MAE': results_plsda['std_mae'],
        'Selected_Features': results_plsda['selected_features_per_fold']
    }
    
    # 4. SVM
    if verbose: 
        print("\n" + "="*70)
        print("4/4: Running SVM (RBF) with Nested CV")
        print("="*70)
    results_svm = nested_cv_with_feature_selection(
        X, y_encoded, y_raw, k_features, model_type='svm', verbose=verbose
    )
    all_results['SVM'] = {
        'Model': 'SVM (RBF)',
        'Mean_Accuracy': results_svm['mean_accuracy'],
        'Std_Accuracy': results_svm['std_accuracy'],
        'MAE': results_svm['mean_mae'],
        'Std_MAE': results_svm['std_mae'],
        'Selected_Features': results_svm['selected_features_per_fold']
    }
    
    # Consolidate results
    comparison_df = pd.DataFrame([
        {
            'Model': v['Model'],
            'Mean_Accuracy': v['Mean_Accuracy'],
            'Std_Accuracy': v['Std_Accuracy'],
            'MAE': v['MAE'],
            'Std_MAE': v['Std_MAE']
        }
        for v in all_results.values()
    ])
    
    if verbose:
        print("\n" + "="*70)
        print(f"CONSOLIDATED MODEL COMPARISON (k={k_features}, Nested CV)")
        print("="*70)
        print(comparison_df.sort_values(by='Mean_Accuracy', ascending=False).to_string(index=False))
    
    comparison_df.to_csv(os.path.join(results_dirs['data'], 'nested_cv_model_comparison.csv'), index=False)
    
    # Plot comparison
    try:
        comparison_plot_data = comparison_df.rename(columns={
            'Mean_Accuracy': 'Accuracy', 
            'Std_Accuracy': 'Accuracy_Std',
            'MAE': 'MAE', 
            'Std_MAE': 'MAE_Std'
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot
        models = comparison_plot_data['Model']
        x_pos = np.arange(len(models))
        axes[0].bar(x_pos, comparison_plot_data['Accuracy'], 
                   yerr=comparison_plot_data['Accuracy_Std'],
                   capsize=5, color='skyblue', edgecolor='black')
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy Comparison', fontsize=12)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # MAE plot
        axes[1].bar(x_pos, comparison_plot_data['MAE'],
                   yerr=comparison_plot_data['MAE_Std'],
                   capsize=5, color='salmon', edgecolor='black')
        axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[1].set_title('Model MAE Comparison', fontsize=12)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'Nested CV Model Comparison (k={k_features} features, LOOCV)', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(results_dirs['figures'], 'nested_cv_model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not generate plot. Error: {e}")
    
    return all_results


def main():
    """Run complete classification workflow with nested CV."""
    
    # Configuration
    DATA_FILE = "data/badata.csv"
    GROUP_COL = "Groups"
    GROUP_ORDER = ["Green", "Ripe", "Overripe"]
    K_FEATURES = 3
    
    print("\n" + "="*70)
    print(f"CV WORKFLOW (k={K_FEATURES} Features)")
    print("Feature selection happens INSIDE each CV fold")
    print("="*70 + "\n")
    
    results_dirs = create_results_directory(
        base_path='results', 
        run_name=f'corrected_nested_cv_k{K_FEATURES}'
    )
    
    # Load and prepare data
    print("STEP 1: Loading and preparing data...")
    df = load_and_impute(DATA_FILE, group_col=GROUP_COL, impute=True, 
                         fill_value=0, drop_threshold=0.35)
    df = df.drop(columns='Unnamed: 0', errors='ignore')
    df = set_categorical_order(df, col=GROUP_COL, categories=GROUP_ORDER)
    X_full, y_raw, _, _, _ = prepare_data(df, target_col=GROUP_COL, drop_missing=True)
    X_full = X_full.select_dtypes(include=[np.number])
    
    # Handle negative values
    if (X_full < 0).any().any():
        for col in X_full.columns:
            if (X_full[col] < 0).any():
                X_full[col] = X_full[col] + abs(X_full[col].min()) + 1
    
    y_encoded = np.array(encode_ordinal_target(y_raw, categories=GROUP_ORDER))
    
    print(f"Total features: {X_full.shape[1]}, Samples: {X_full.shape[0]}")
    print(f"Will select top k={K_FEATURES} features WITHIN each CV fold\n")
    
    # Run nested CV
    print("="*70)
    print("STEP 2: Running Nested CV Model Comparison")
    print("="*70)
    
    model_results = run_nested_model_comparison(
        X_full, y_encoded, y_raw, K_FEATURES, results_dirs, verbose=True
    )
    
    # Save results
    print("\n" + "="*70)
    print("STEP 3: Saving Results")
    print("="*70)
    
    summary = {
        'k_features': K_FEATURES,
        'n_samples': X_full.shape[0],
        'cv_strategy': 'LOOCV with nested feature selection',
        'model_results': {k: {
            'Model': v['Model'],
            'Mean_Accuracy': float(v['Mean_Accuracy']),
            'Std_Accuracy': float(v['Std_Accuracy']),
            'MAE': float(v['MAE']),
            'Std_MAE': float(v['Std_MAE'])
        } for k, v in model_results.items()}
    }
    
    save_results(summary, os.path.join(results_dirs['base'], 'nested_cv_results_summary.json'))
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Feature Count (k): {K_FEATURES} (selected independently in each fold)")
    print(f"CV Strategy: LOOCV")
    print(f"Sample Size: {X_full.shape[0]}")
    
    best_model_key = max(model_results.keys(), 
                        key=lambda k: model_results[k]['Mean_Accuracy'])
    best_model = model_results[best_model_key]
    
    print(f"\nBest Model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Mean_Accuracy']:.3f} ± {best_model['Std_Accuracy']:.3f}")
    print(f"MAE: {best_model['MAE']:.3f} ± {best_model['Std_MAE']:.3f}")
    
    print(f"\nResults saved in: {results_dirs['base']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()