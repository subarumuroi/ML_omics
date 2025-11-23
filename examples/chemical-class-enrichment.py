"""
Chemical Class & Pathway Enrichment Analysis for Ordinal Regression Features
=============================================================================
Performs hypergeometric test to identify overrepresented chemical classes
or pathways in your top-ranked features from ordinal regression/ANOVA.
"""

import pandas as pd
import numpy as np
from scipy.stats import hypergeom, fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

class EnrichmentAnalyzer:
    """
    Performs enrichment analysis on ranked features from ML models.
    """
    
    def __init__(self, compound_database):
        """
        Parameters:
        -----------
        compound_database : dict
            Database mapping feature names to their properties
            Format: {feature_name: {'class': ..., 'pathway': ..., 'subclass': ...}}
        """
        self.database = compound_database
        self._build_annotations()
    
    def _build_annotations(self):
        """Build annotation mappings from database"""
        self.feature_to_class = {}
        self.feature_to_pathway = {}
        self.feature_to_subclass = {}
        
        for feature, info in self.database.items():
            self.feature_to_class[feature] = info.get('class', 'Unknown')
            self.feature_to_pathway[feature] = info.get('pathway', 'Unknown')
            self.feature_to_subclass[feature] = info.get('subclass', 'Unknown')
    
    def hypergeometric_enrichment(self, selected_features, all_features, 
                                   annotation_type='class', min_category_size=2):
        """
        Perform hypergeometric test for enrichment.
        
        Parameters:
        -----------
        selected_features : list
            Your top N features (e.g., from ANOVA, feature importance)
        all_features : list
            All features in your dataset (background set)
        annotation_type : str
            'class', 'pathway', or 'subclass'
        min_category_size : int
            Minimum number of features in a category to test
            
        Returns:
        --------
        pd.DataFrame with enrichment results
        """
        # Get annotation mapping
        if annotation_type == 'class':
            annotation_map = self.feature_to_class
        elif annotation_type == 'pathway':
            annotation_map = self.feature_to_pathway
        elif annotation_type == 'subclass':
            annotation_map = self.feature_to_subclass
        else:
            raise ValueError("annotation_type must be 'class', 'pathway', or 'subclass'")
        
        # Build annotation sets
        selected_annotated = [f for f in selected_features if f in annotation_map]
        all_annotated = [f for f in all_features if f in annotation_map]
        
        # Get unique categories
        categories = set(annotation_map.values()) - {'Unknown'}
        
        results = []
        
        for category in categories:
            # Count features in this category
            selected_in_cat = sum(1 for f in selected_annotated 
                                  if annotation_map.get(f) == category)
            all_in_cat = sum(1 for f in all_annotated 
                            if annotation_map.get(f) == category)
            
            # Skip small categories
            if all_in_cat < min_category_size:
                continue
            
            # Hypergeometric test parameters
            M = len(all_annotated)  # Total population
            n = all_in_cat  # Total successes in population
            N = len(selected_annotated)  # Sample size
            k = selected_in_cat  # Observed successes in sample
            
            # P-value: probability of observing k or more successes
            pval = hypergeom.sf(k - 1, M, n, N)
            
            # Calculate enrichment metrics
            expected = (n / M) * N
            fold_enrichment = (k / N) / (n / M) if n > 0 else 0
            
            results.append({
                'Category': category,
                'Type': annotation_type,
                'Selected_Count': k,
                'Selected_Total': N,
                'Background_Count': n,
                'Background_Total': M,
                'Expected': expected,
                'Fold_Enrichment': fold_enrichment,
                'P_value': pval
            })
        
        # Convert to DataFrame and adjust p-values
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            # Multiple testing correction (Benjamini-Hochberg)
            df['P_adjusted'] = multipletests(df['P_value'], method='fdr_bh')[1]
            df['Significant'] = df['P_adjusted'] < 0.05
            
            # Sort by p-value
            df = df.sort_values('P_value')
        
        return df
    
    def fisher_exact_enrichment(self, selected_features, all_features, 
                                annotation_type='class', min_category_size=2):
        """
        Alternative: Fisher's exact test (better for small sample sizes)
        """
        if annotation_type == 'class':
            annotation_map = self.feature_to_class
        elif annotation_type == 'pathway':
            annotation_map = self.feature_to_pathway
        else:
            annotation_map = self.feature_to_subclass
        
        selected_annotated = [f for f in selected_features if f in annotation_map]
        all_annotated = [f for f in all_features if f in annotation_map]
        
        categories = set(annotation_map.values()) - {'Unknown'}
        results = []
        
        for category in categories:
            # Build contingency table
            selected_in_cat = sum(1 for f in selected_annotated 
                                  if annotation_map.get(f) == category)
            selected_not_cat = len(selected_annotated) - selected_in_cat
            
            all_in_cat = sum(1 for f in all_annotated 
                            if annotation_map.get(f) == category)
            
            if all_in_cat < min_category_size:
                continue
                
            bg_in_cat = all_in_cat - selected_in_cat
            bg_not_cat = len(all_annotated) - len(selected_annotated) - bg_in_cat
            
            # 2x2 contingency table
            table = [[selected_in_cat, selected_not_cat],
                     [bg_in_cat, bg_not_cat]]
            
            oddsratio, pval = fisher_exact(table, alternative='greater')
            
            results.append({
                'Category': category,
                'Type': annotation_type,
                'Selected_Count': selected_in_cat,
                'Selected_Total': len(selected_annotated),
                'Background_Count': all_in_cat,
                'Odds_Ratio': oddsratio,
                'P_value': pval
            })
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            df['P_adjusted'] = multipletests(df['P_value'], method='fdr_bh')[1]
            df['Significant'] = df['P_adjusted'] < 0.05
            df = df.sort_values('P_value')
        
        return df
    
    def plot_enrichment(self, enrichment_df, top_n=15, figsize=(10, 6)):
        """
        Create enrichment visualization
        """
        # Take top N results
        plot_df = enrichment_df.head(top_n).copy()
        
        # Calculate -log10(p-value) for visualization
        plot_df['-log10(P_adj)'] = -np.log10(plot_df['P_adjusted'])
        
        # Color by significance
        colors = ['#e74c3c' if sig else '#95a5a6' 
                  for sig in plot_df['Significant']]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(plot_df)), plot_df['-log10(P_adj)'], 
                       color=colors, alpha=0.8)
        
        # Add significance threshold line
        threshold = -np.log10(0.05)
        ax.axvline(threshold, color='black', linestyle='--', 
                   linewidth=1, alpha=0.5, label='p=0.05')
        
        # Formatting
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['Category'])
        ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        ax.set_title('Enrichment Analysis Results', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Add fold enrichment as text
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            if 'Fold_Enrichment' in row:
                ax.text(row['-log10(P_adj)'] + 0.1, i, 
                       f"×{row['Fold_Enrichment']:.2f}",
                       va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def enrichment_summary_table(self, enrichment_df):
        """
        Create a publication-ready summary table
        """
        summary_cols = [
            'Category', 
            'Selected_Count', 
            'Selected_Total',
            'Background_Count',
            'Background_Total',
            'Fold_Enrichment',
            'P_value',
            'P_adjusted'
        ]
        
        # Select and rename columns
        table = enrichment_df[summary_cols].copy()
        table.columns = [
            'Category',
            'In Selection',
            'Total Selected',
            'In Background',
            'Total Background',
            'Fold Enrichment',
            'P-value',
            'Adj. P-value'
        ]
        
        # Format numbers
        table['Fold Enrichment'] = table['Fold Enrichment'].round(2)
        table['P-value'] = table['P-value'].apply(lambda x: f'{x:.2e}')
        table['Adj. P-value'] = table['Adj. P-value'].apply(lambda x: f'{x:.2e}')
        
        return table


# =============================================================================
# Example Usage with Your Workflow
# =============================================================================

if __name__ == "__main__":
    # Example: Mock compound database (replace with your actual database)
    COMPOUND_DATABASE = {
        'Ethyl 3-methylbutanoate Area': {
            'class': 'Esters',
            'pathway': 'Ester biosynthesis',
            'subclass': 'Branched-chain esters'
        },
        '2-Methylbutyl acetate Area': {
            'class': 'Esters',
            'pathway': 'Ester biosynthesis',
            'subclass': 'Acetate esters'
        },
        'trans-2-Penten-1-ol Area': {
            'class': 'Alcohols',
            'pathway': 'Fatty acid degradation',
            'subclass': 'Unsaturated alcohols'
        },
        '3-Hexanone Area': {
            'class': 'Ketones',
            'pathway': 'Fatty acid degradation',
            'subclass': 'Methyl ketones'
        },
        'Allyl isothiocyanate Area': {
            'class': 'Isothiocyanates',
            'pathway': 'Glucosinolate degradation',
            'subclass': 'Aliphatic isothiocyanates'
        },
        # ... add all your compounds
    }
    
    # Simulate your workflow
    # 1. Top 10 features from ANOVA/feature importance
    top_features = [
        'Ethyl 3-methylbutanoate Area',
        '2-Methylbutyl acetate Area',
        'Isoamyl acetate Area',
        'trans-2-Penten-1-ol Area',
        '(Z)-2-Penten-1-ol Area',
        'Propyl acetate Area',
        '3-Hexanone Area',
        '3-Methyl-2-buten-1-ol Area',
        'Methyl isovalerate Area',
        'Allyl isothiocyanate Area'
    ]
    
    # 2. All features in your dataset (background)
    all_features = top_features + [
        # Add the rest of your features...
        'Feature_11', 'Feature_12', # etc.
    ]
    
    # 3. Initialize enrichment analyzer
    analyzer = EnrichmentAnalyzer(COMPOUND_DATABASE)
    
    # 4. Run enrichment analysis
    print("Running Chemical Class Enrichment...")
    class_enrichment = analyzer.hypergeometric_enrichment(
        selected_features=top_features,
        all_features=all_features,
        annotation_type='class'
    )
    
    print("\n" + "="*70)
    print("CHEMICAL CLASS ENRICHMENT RESULTS")
    print("="*70)
    print(class_enrichment)
    
    # 5. Run pathway enrichment
    print("\n\nRunning Pathway Enrichment...")
    pathway_enrichment = analyzer.hypergeometric_enrichment(
        selected_features=top_features,
        all_features=all_features,
        annotation_type='pathway'
    )
    
    print("\n" + "="*70)
    print("PATHWAY ENRICHMENT RESULTS")
    print("="*70)
    print(pathway_enrichment)
    
    # 6. Visualize
    fig = analyzer.plot_enrichment(class_enrichment)
    plt.savefig('class_enrichment.png', dpi=300, bbox_inches='tight')
    
    # 7. Create publication table
    table = analyzer.enrichment_summary_table(class_enrichment)
    table.to_csv('enrichment_results.csv', index=False)
    print("\n\nPublication-ready table saved to 'enrichment_results.csv'")