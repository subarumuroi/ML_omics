"""
General Classification Workflow for Metabolomics Data (Badata).

Demonstrative version: Uses a fixed feature set (k=10) and Leave-One-Out Cross-Validation (LOOCV)
to highlight model application on small datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import package modules
from preprocessing import (
    load_and_impute,
    select_k_best_features, 
)
from models.ordinal import (
    encode_ordinal_target,
    train_evaluate_ordinal, 
)
from models import train_evaluate_rf 

# Imports for PLS-DA and SVM
from models.pls_da import train_evaluate_plsda
from models.svm import train_evaluate_svm


from utils import (
    prepare_data,
    set_categorical_order,
    create_results_directory,
    save_results,
)
from visualization.ordinal_plots import (
    plot_ordinal_model_comparison,
)

# --- CORE FUNCTION: Modular Model Comparison ---

def run_model_comparison(X, y_encoded, y_raw, optimal_k, results_dirs, verbose=True, N_SPLITS=9):
    """
    Runs four different classification models on the given feature set and
    consolidates their performance metrics (Accuracy, MAE, and Stds).
    
    Parameters
    ----------
    ...
    N_SPLITS : int, default=9
        The consistent number of splits to use for cross-validation (e.g., 9 for LOOCV).
    """
    
    all_results = {}
    
    # ----------------------------------------------------------------------
    # 1. Ordinal Regression (LogisticAT) - Explicitly passing N_SPLITS
    # ----------------------------------------------------------------------
    if verbose: print("\n--- 1/4: Running Ordinal Regression (LogisticAT) ---")
    results_ord = train_evaluate_ordinal(
        X, y_encoded, model_type='LogisticAT', scaler='minmax', log_transform=True, n_splits=N_SPLITS, verbose=verbose
    )
    all_results['Ordinal_LogisticAT'] = {
        'Model': 'Ordinal (LogAT)',
        'Mean_Accuracy': results_ord['mean_accuracy'],
        'Std_Accuracy': results_ord['std_accuracy'],
        'MAE': results_ord['mean_mae'],
        'Std_MAE': results_ord['std_mae'],
    }
    
    # ----------------------------------------------------------------------
    # 2. Random Forest (RF) - Explicitly passing N_SPLITS
    # ----------------------------------------------------------------------
    if verbose: print("\n--- 2/4: Running Random Forest (RF) ---")
    results_rf_obj = train_evaluate_rf(X, y_encoded, n_splits=N_SPLITS, verbose=verbose)
    
    # Access metrics via the .cv_results attribute
    rf_metrics = results_rf_obj.cv_results
    
    all_results['Random_Forest'] = {
        'Model': 'Random Forest',
        'Mean_Accuracy': rf_metrics.get('mean_accuracy', np.nan),
        'Std_Accuracy': rf_metrics.get('std_accuracy', np.nan),
        'MAE': np.nan, 
        'Std_MAE': np.nan,
    }
    
    # ----------------------------------------------------------------------
    # 3. PLS-DA - Explicitly passing N_SPLITS
    # ----------------------------------------------------------------------
    if verbose: print("\n--- 3/4: Running PLS-DA ---")
    results_plsda = train_evaluate_plsda(X, y_raw, n_splits=N_SPLITS, verbose=verbose)
    all_results['PLS_DA'] = {
        'Model': 'PLS-DA',
        'Mean_Accuracy': results_plsda['mean_accuracy'],
        'Std_Accuracy': results_plsda['std_accuracy'],
        'MAE': results_plsda['mean_mae'],
        'Std_MAE': results_plsda.get('std_mae', np.nan),
    }
    
    # ----------------------------------------------------------------------
    # 4. SVM - Explicitly passing N_SPLITS
    # ----------------------------------------------------------------------
    if verbose: print("\n--- 4/4: Running SVM (RBF) ---")
    results_svm = train_evaluate_svm(X, y_encoded, n_splits=N_SPLITS, verbose=verbose)
    all_results['SVM'] = {
        'Model': 'SVM (RBF)',
        'Mean_Accuracy': results_svm['mean_accuracy'],
        'Std_Accuracy': results_svm['std_accuracy'],
        'MAE': results_svm['mean_mae'],
        'Std_MAE': results_svm.get('std_mae', np.nan),
    }
    
    # === Consolidation and Output ===
    comparison_df = pd.DataFrame(list(all_results.values()))
    
    if verbose:
        print("\n" + "="*50)
        print(f"CONSOLIDATED MODEL COMPARISON (k={optimal_k})")
        print("="*50)
        print(comparison_df.sort_values(by='Mean_Accuracy', ascending=False).to_markdown(index=False))

    comparison_df.to_csv(os.path.join(results_dirs['data'], 'general_model_comparison.csv'), index=False)
    
    # Visualization: Plotting the comparison
    try:
        comparison_plot_data = comparison_df.rename(columns={
            'Mean_Accuracy': 'Accuracy', 
            'Std_Accuracy': 'Accuracy_Std',
            'MAE': 'MAE', 
            'Std_MAE': 'MAE_Std'
        }).reset_index(drop=True)
        
        fig, _ = plot_ordinal_model_comparison(comparison_plot_data)
        fig.suptitle(f'Model Comparison (Fixed k={optimal_k} features, LOOCV)', fontsize=14)
        fig.savefig(os.path.join(results_dirs['figures'], 'general_model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not generate model comparison plot. Error: {e}")

    return all_results

# --- MAIN WORKFLOW (Revised) ---

def main():
    """Run complete general classification workflow."""
    
    # Configuration
    DATA_FILE = "data/badata.csv"
    GROUP_COL = "Groups"
    GROUP_ORDER = ["Green", "Ripe", "Overripe"]
    
    # NEW CONFIGURATION: FIXED FEATURE COUNT AND LOOCV
    # We fix k=10 for demonstration and use LOOCV for the small dataset (N=9 samples)
    DEMO_K_FEATURES = 10
    LOOCV_SPLITS = 9 # N=9 samples, so LOOCV is N splits
    
    print("\n" + "="*70)
    print(f"GENERAL CLASSIFICATION WORKFLOW (k={DEMO_K_FEATURES} Features, LOOCV)")
    print("======================================================================" + "\n")
    
    results_dirs = create_results_directory(base_path='results', run_name=f'general_badata_k{DEMO_K_FEATURES}_loocv_analysis')
    
    # STEP 1: Load, Prepare, and Clean Data
    print("STEP 1: Loading and preparing data...")
    df = load_and_impute(DATA_FILE, group_col=GROUP_COL, impute=True, fill_value=0, drop_threshold=0.35)
    df = df.drop(columns='Unnamed: 0', errors='ignore')
    df = set_categorical_order(df, col=GROUP_COL, categories=GROUP_ORDER)
    X_full, y_raw, _, _, _ = prepare_data(df, target_col=GROUP_COL, drop_missing=True)
    X_full = X_full.select_dtypes(include=[np.number])
    
    # Handle negative values for log transformation
    if (X_full < 0).any().any():
        for col in X_full.columns:
            if (X_full[col] < 0).any():
                X_full[col] = X_full[col] + abs(X_full[col].min()) + 1
    
    y_encoded = np.array(encode_ordinal_target(y_raw, categories=GROUP_ORDER))
    
    print(f"Initial features: {X_full.shape[1]}, Samples: {X_full.shape[0]}")
    
    # ======================================================================
    # STEP 2: Feature Selection (Simplified for Demonstration)
    # ======================================================================
    print("\n" + "="*70)
    print(f"STEP 2: Selecting Top k={DEMO_K_FEATURES} Features (Skipping Optimal Search)")
    print("======================================================================")
    
    if DEMO_K_FEATURES > X_full.shape[1]:
        print(f"Warning: k={DEMO_K_FEATURES} exceeds total features. Using all {X_full.shape[1]} features.")
        X_optimal = X_full
        selected_features = X_full.columns.tolist()
    else:
        # Select the top K features based on OR/ANOVA ranking
        # This still uses the feature ranking logic but we fix the output size k
        X_optimal_data, selected_features, _ = select_k_best_features(X_full, y_raw, k=DEMO_K_FEATURES, verbose=True)
        X_optimal = pd.DataFrame(X_optimal_data, columns=selected_features)
    
    print(f"✓ Selected {X_optimal.shape[1]} features: {', '.join(selected_features[:5])}... ({len(selected_features) - 5} more)")


    # ======================================================================
    # STEP 3: Run Model Comparison on Fixed Feature Set (Using LOOCV)
    # ======================================================================
    print("\n" + "="*70)
    print(f"STEP 3: Running 4 Models on k={X_optimal.shape[1]} Features with LOOCV (N_SPLITS={LOOCV_SPLITS})")
    print("======================================================================")
    
    model_comparison_results = run_model_comparison(
        X_optimal, y_encoded, y_raw, X_optimal.shape[1], results_dirs, verbose=True, N_SPLITS=LOOCV_SPLITS
    )

    # STEP 4: Save Summary Results
    print("\n" + "="*70)
    print("STEP 4: Saving Results")
    print("======================================================================")
    
    summary = {
        'optimal_k_used': X_optimal.shape[1],
        'n_splits_used': LOOCV_SPLITS,
        'selected_features': selected_features,
        'model_comparison_results': model_comparison_results,
    }
    
    save_results(summary, os.path.join(results_dirs['base'], 'general_results_summary.json'))
    
    # Final Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("======================================================================")
    print(f"Feature Count Used (k): {X_optimal.shape[1]}")
    print(f"Cross-Validation Strategy: LOOCV (N={LOOCV_SPLITS})")
    
    best_model = max(model_comparison_results.values(), key=lambda x: x['Mean_Accuracy'])
    print(f"\nBest Classification Model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Mean_Accuracy']:.3f}±{best_model['Std_Accuracy']:.3f}")
    print(f"MAE: {best_model['MAE']:.3f}")
    print(f"\nResults and comparison plots saved in: {results_dirs['base']}")
    print("======================================================================" + "\n")


if __name__ == "__main__":
    main()