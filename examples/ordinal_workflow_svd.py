"""
Ordinal Regression Workflow with SVD-Guided Features (Per Omics Type)

Demonstrates:
1. Independent SVD decomposition per metabolomics dataset
2. Projection onto top components
3. Ordinal regression using LogisticAT
4. Heatmap visualization of SVD loadings

Requirements: Your ML_omics package with mord, scikit-learn, pandas, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_impute, select_k_best_features
from utils import prepare_data, create_results_directory, save_results
from models.ordinal import train_evaluate_ordinal, encode_ordinal_target
from visualization.ordinal_plots import plot_ordinal_coefficients

# -----------------------------
# CONFIG
# -----------------------------
FILES = {
    'AminoAcids': 'data/badata-amino-acids.csv',
    'Aromatics': 'data/badata-aromatics.csv',
    'Metabolomics': 'data/badata-metabolomics.csv'
}

GROUP_COL = 'Groups'
GROUP_ORDER = ['Green', 'Ripe', 'Overripe']
VAR_THRESHOLD = 0.8  # Cumulative explained variance for SVD
TOP_COMPONENTS_FOR_PLOT = 3  # How many components to show in heatmap

RESULTS_DIR = create_results_directory(base_path='results', run_name='ordinal_svd_per_omics')

# -----------------------------
# STEP 1: LOAD & PREPROCESS
# -----------------------------
data_dict = {}
for name, path in FILES.items():
    df = load_and_impute(path, group_col=GROUP_COL, impute=True, fill_value=0)
    data_dict[name] = df

# -----------------------------
# STEP 2: SVD + ORDINAL REGRESSION PER DATASET
# -----------------------------
for name, df in data_dict.items():
    print(f"\n===== Processing {name} dataset =====")
    X = df.select_dtypes(include=[np.number])

    # Center & scale per feature
    X_scaled = (X - X.mean()) / X.std(ddof=0)

    # Perform SVD
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

    # Explained variance
    explained_var = S**2 / np.sum(S**2)
    cum_var = np.cumsum(explained_var)
    k = np.searchsorted(cum_var, VAR_THRESHOLD) + 1
    print(f"Selected top {k} components to explain {VAR_THRESHOLD*100:.0f}% variance.")

    # Project samples onto top k components
    X_svd = U[:, :k] @ np.diag(S[:k])
    X_svd_df = pd.DataFrame(X_svd, index=df.index,
                            columns=[f'{name}_Comp{i+1}' for i in range(k)])

    # Encode ordinal target
    _, y_raw, _, _, _ = prepare_data(df, target_col=GROUP_COL)
    y = encode_ordinal_target(y_raw, categories=GROUP_ORDER)

    # Scale components for modeling
    scaler = MinMaxScaler()
    X_scaled_final = pd.DataFrame(scaler.fit_transform(X_svd_df), columns=X_svd_df.columns)

    # Optional: Feature selection (leave as all components for now)
    selected_features = X_scaled_final.columns.tolist()
    print(f"Using {len(selected_features)} components for ordinal regression.")

    # Train ordinal regression
    results = train_evaluate_ordinal(X_scaled_final, y, log_transform=False)
    print(f"Ordinal Regression Performance for {name}:")
    print(f"  Accuracy: {results['mean_accuracy']:.3f}")
    print(f"  MAE: {results['mean_mae']:.3f}")

    # -----------------------------
    # STEP 3: VISUALIZE COEFFICIENTS
    # -----------------------------
    fig, _ = plot_ordinal_coefficients(results['coefficients'],
                                       top_n=min(10, len(selected_features)))
    fig.savefig(RESULTS_DIR['figures'] / f'{name}_ordinal_svd_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------------
    # STEP 4: VISUALIZE SVD LOADINGS
    # -----------------------------
    loadings_df = pd.DataFrame(Vt[:TOP_COMPONENTS_FOR_PLOT, :].T, index=X.columns,
                               columns=[f'{name}_Comp{i+1}' for i in range(TOP_COMPONENTS_FOR_PLOT)])
    plt.figure(figsize=(8,6))
    sns.heatmap(loadings_df, cmap='vlag', center=0)
    plt.title(f"{name} SVD Loadings (Top {TOP_COMPONENTS_FOR_PLOT} Components)")
    plt.xlabel("Components")
    plt.ylabel("Original Features")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR['figures'] / f"{name}_svd_loadings.png", dpi=300)
    plt.close()

    # -----------------------------
    # STEP 5: SAVE RESULTS
    # -----------------------------
    loadings_df.to_csv(RESULTS_DIR['data'] / f'{name}_svd_loadings.csv')

    summary = {
        'dataset': name,
        'n_samples': df.shape[0],
        'n_features_original': X.shape[1],
        'n_components': X_scaled_final.shape[1],
        'explained_variance_threshold': VAR_THRESHOLD,
        'cv_accuracy': results['mean_accuracy'],
        'cv_mae': results['mean_mae'],
        'selected_components': selected_features
    }

    save_results(summary, RESULTS_DIR['data'] / f'{name}_ordinal_svd_results.json')

print("\n✓ Workflow complete for all datasets. Results saved in:", RESULTS_DIR['base'])
# -----------------------------
# STEP 6: CROSS-OMICS FEATURE CONTRIBUTION PLOT
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

TOP_COMPONENTS_FOR_PLOT = 2  # Use same as ordinal regression

feature_contrib = []

for name, df in data_dict.items():
    X = df.select_dtypes(include=[np.number])
    X_scaled = (X - X.mean()) / X.std(ddof=0)
    _, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    
    # Take top N components
    for i in range(TOP_COMPONENTS_FOR_PLOT):
        comp_name = f"{name}_Comp{i+1}"
        for feat_idx, feat in enumerate(X.columns):
            feature_contrib.append({
                'Dataset': name,
                'Component': comp_name,
                'Feature': feat,
                'Loading': Vt[i, feat_idx]
            })

contrib_df = pd.DataFrame(feature_contrib)

# Pivot for heatmap
heatmap_df = contrib_df.pivot(index='Feature', columns=['Dataset','Component'], values='Loading')

plt.figure(figsize=(12, max(6, len(heatmap_df)/2)))
sns.heatmap(heatmap_df, cmap='vlag', center=0)
plt.title("Cross-Omics Feature Contributions (Top Components)")
plt.xlabel("Dataset and Component")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(RESULTS_DIR['figures'] / "cross_omics_feature_contributions.png", dpi=300)
plt.close()

print("✓ Cross-omics feature contribution plot saved: cross_omics_feature_contributions.png")
