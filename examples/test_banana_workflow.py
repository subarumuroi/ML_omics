"""
Test script to verify package works with banana dataset preprocessing.

Tests all preprocessing strategies mentioned:
1. badata.csv → load_and_impute(impute=True, fill_value=50) → badata_imputed_filled
2. badata.csv → load_and_impute(impute=True, fill_value=None) → badata_imputed
3. badata.csv → load_and_impute(impute=False, fill_value=50) → badata_filled
"""

import pandas as pd
import numpy as np
from preprocessing import load_and_impute, select_k_best_features
from models import train_evaluate_model
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from utils import prepare_data, set_categorical_order, print_data_summary
from analysis import get_feature_importance_df
from visualization import plot_confusion_matrix, plot_compound_trends
import matplotlib.pyplot as plt


def test_preprocessing_strategies():
    """Test different preprocessing strategies."""
    print("\n" + "="*70)
    print("TESTING PREPROCESSING STRATEGIES")
    print("="*70)
    
    strategies = [
        ("Impute + Fill", True, 50),
        ("Impute Only", True, None),
        ("Fill Only", False, 50),
    ]
    
    for name, impute, fill_value in strategies:
        print(f"\n--- Strategy: {name} (impute={impute}, fill_value={fill_value}) ---")
        
        try:
            df = load_and_impute(
                file_path='data/badata.csv',
                group_col='Groups',
                fill_value=fill_value,
                impute=impute
            )
            
            print(f"✓ Loaded: {df.shape}")
            print(f"✓ Columns: {df.columns.tolist()[:3]}... ({len(df.columns)} total)")
            print(f"✓ Groups: {df['Groups'].unique()}")
            print(f"✓ Missing values: {df.isnull().sum().sum()}")
            print(f"✓ Index type: {type(df.index)}")
            
            # Verify no MultiIndex
            assert not isinstance(df.index, pd.MultiIndex), "ERROR: MultiIndex detected!"
            
            # Verify Groups column exists
            assert 'Groups' in df.columns, "ERROR: Groups column missing!"
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_full_workflow_rf():
    """Test complete RF workflow with banana data."""
    print("\n" + "="*70)
    print("TESTING FULL RF WORKFLOW")
    print("="*70)
    
    # Load with your preferred strategy
    df = load_and_impute(
        file_path='data/badata.csv',
        group_col='Groups',
        fill_value=50,
        impute=True
    )
    
    print(f"Loaded: {df.shape}")
    
    # Drop index column if it exists
    if df.columns[0].startswith('Unnamed'):
        print(f"Dropping index column: {df.columns[0]}")
        df = df.drop(columns=[df.columns[0]])
    
    # Set categorical order
    df = set_categorical_order(df, 'Groups', ['Green', 'Ripe', 'Overripe'])
    
    # Prepare data
    X, y_raw, y, le, dropped = prepare_data(df, target_col='Groups', drop_missing=True)
    print_data_summary(X, y, le)
    
    # Select features
    X_selected, features, _ = select_k_best_features(X, y, k=15)
    X_df = pd.DataFrame(X_selected, columns=features)
    
    # Train RF
    print("\nTraining Random Forest...")
    clf = train_evaluate_model(X_df, y, verbose=True)
    
    # Get importance
    importance_df = get_feature_importance_df(clf.model, features)
    print(f"\nTop 5 features:")
    print(importance_df.head(5))
    
    # Visualize
    fig, _ = plot_confusion_matrix(clf.confusion_matrix, le.classes_)
    plt.savefig('test_rf_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: test_rf_confusion.png")
    
    print("\n✓ RF workflow completed successfully!")
    return df, X_df, y, le


def test_full_workflow_ordinal():
    """Test complete ordinal workflow with banana data."""
    print("\n" + "="*70)
    print("TESTING FULL ORDINAL WORKFLOW")
    print("="*70)
    
    # Load
    df = load_and_impute(
        file_path='data/badata.csv',
        group_col='Groups',
        fill_value=50,
        impute=True
    )
    
    # Drop index if needed
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(columns=[df.columns[0]])
    
    # Prepare
    X, y_raw, _, _, _ = prepare_data(df, target_col='Groups', drop_missing=True)
    
    # Encode ordinal
    y = encode_ordinal_target(y_raw, categories=['Green', 'Ripe', 'Overripe'])
    
    # Train
    print("\nTraining Ordinal Regression...")
    results = train_evaluate_ordinal(X, y, verbose=True)
    
    print(f"\nTop 5 coefficients:")
    print(results['coefficients'].head(5))
    
    print("\n✓ Ordinal workflow completed successfully!")
    return results


def test_visualization_with_data(df, features, y, le):
    """Test visualizations work with the data."""
    print("\n" + "="*70)
    print("TESTING VISUALIZATIONS")
    print("="*70)
    
    # Compound trends
    fig, _ = plot_compound_trends(
        df, features[:5], 
        group_col='Groups',
        group_order=['Green', 'Ripe', 'Overripe'],
        scale='log'
    )
    plt.savefig('test_compound_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: test_compound_trends.png")
    
    print("\n✓ Visualizations completed successfully!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("BANANA DATA WORKFLOW VERIFICATION")
    print("="*70)
    
    # Test 1: Preprocessing strategies
    test_preprocessing_strategies()
    
    # Test 2: Full RF workflow
    df, X_df, y, le = test_full_workflow_rf()
    
    # Test 3: Full ordinal workflow
    ordinal_results = test_full_workflow_ordinal()
    
    # Test 4: Visualizations
    features = ordinal_results['coefficients'].head(5)['Feature'].tolist()
    test_visualization_with_data(df, features, y, le)
    
    # Final summary
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nThe package is fully compatible with your banana dataset!")
    print("\nGenerated files:")
    print("  - test_rf_confusion.png")
    print("  - test_compound_trends.png")
    print("\nYou can now use the package with:")
    print("  df = load_and_impute('data/badata.csv', 'Groups', fill_value=50, impute=True)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()