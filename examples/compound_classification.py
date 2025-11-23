# compound_classifications.py
# Custom mapping for YOUR specific banana ripening biomarkers
# Created based on your actual top features from ordinal regression analysis

COMPOUND_DATABASE = {
    # ============================================================================
    # AROMATICS / VOLATILES (from GC-MS)
    # ============================================================================
    "Ethyl 3-methylbutanoate Area": {
        "common_name": "Ethyl 3-methylbutanoate",
        "chemical_class": "Esters",
        "subclass": "Branched-chain fatty acid esters",
        "pathway": "Leucine catabolism → ester synthesis",
        "aroma": "Fruity, apple-like",
        "ripening_role": "Major banana aroma compound, increases dramatically during ripening"
    },
    "2-Methylbutyl acetate Area": {
        "common_name": "2-Methylbutyl acetate",
        "chemical_class": "Esters",
        "subclass": "Branched-chain alcohol esters",
        "pathway": "Amino acid catabolism → ester synthesis",
        "aroma": "Fruity, banana-like",
        "ripening_role": "Key ripening ester, contributes to characteristic banana smell"
    },
    "Isoamyl acetate Area": {
        "common_name": "Isoamyl acetate (3-methylbutyl acetate)",
        "chemical_class": "Esters",
        "subclass": "Branched-chain alcohol esters",
        "pathway": "Leucine → isoamyl alcohol → ester",
        "aroma": "Strong banana, pear-like",
        "ripening_role": "THE banana aroma compound - classic ripening marker"
    },
    "trans-2-Penten-1-ol Area": {
        "common_name": "trans-2-Penten-1-ol",
        "chemical_class": "Alcohols",
        "subclass": "Unsaturated alcohols",
        "pathway": "Lipid oxidation",
        "aroma": "Green, plastic-like",
        "ripening_role": "Decreases with ripening (unripe green note)"
    },
    "(Z)-2-Penten-1-ol Area": {
        "common_name": "cis-2-Penten-1-ol",
        "chemical_class": "Alcohols",
        "subclass": "Unsaturated alcohols",
        "pathway": "Lipid oxidation",
        "aroma": "Green, leafy",
        "ripening_role": "Green note, decreases as fruit ripens"
    },
    "Propyl acetate Area": {
        "common_name": "Propyl acetate",
        "chemical_class": "Esters",
        "subclass": "Simple alkyl acetates",
        "pathway": "Fermentation/ester synthesis",
        "aroma": "Pear-like, fruity",
        "ripening_role": "Early ripening ester"
    },
    "3-Hexanone Area": {
        "common_name": "3-Hexanone",
        "chemical_class": "Ketones",
        "subclass": "Aliphatic ketones",
        "pathway": "Lipid oxidation",
        "aroma": "Ethereal, acetone-like",
        "ripening_role": "Oxidative metabolism marker"
    },
    "3-Methyl-2-buten-1-ol Area": {
        "common_name": "3-Methyl-2-buten-1-ol (prenol)",
        "chemical_class": "Alcohols",
        "subclass": "Unsaturated alcohols",
        "pathway": "Isoprenoid metabolism",
        "aroma": "Herbaceous, green",
        "ripening_role": "Terpenoid precursor"
    },
    "Methyl isovalerate Area": {
        "common_name": "Methyl isovalerate",
        "chemical_class": "Esters",
        "subclass": "Branched-chain fatty acid esters",
        "pathway": "Leucine/valine catabolism",
        "aroma": "Fruity, apple-like",
        "ripening_role": "Branched-chain amino acid degradation product"
    },
    "Allyl isothiocyanate Area": {
        "common_name": "Allyl isothiocyanate",
        "chemical_class": "Isothiocyanates",
        "subclass": "Glucosinolate breakdown products",
        "pathway": "Glucosinolate hydrolysis (stress response)",
        "aroma": "Pungent, mustard-like",
        "ripening_role": "Defense compound, may indicate stress/damage"
    },
    
    # ============================================================================
    # AMINO ACIDS (from LC-MS)
    # ============================================================================
    "ASP": {
        "common_name": "Aspartic acid (Aspartate)",
        "chemical_class": "Amino acids",
        "subclass": "Acidic amino acids",
        "pathway": "Amino acid metabolism / TCA cycle",
        "function": "Neurotransmitter precursor, protein building block",
        "ripening_role": "Decreases - converted to other amino acids and metabolites during ripening"
    },
    "GABA": {
        "common_name": "γ-Aminobutyric acid (GABA)",
        "chemical_class": "Amino acids",
        "subclass": "Non-protein amino acids",
        "pathway": "Glutamate decarboxylation (GABA shunt)",
        "function": "Neurotransmitter, stress signaling",
        "ripening_role": "Increases - stress/signaling compound during ripening senescence"
    },
    "VAL": {
        "common_name": "Valine",
        "chemical_class": "Amino acids",
        "subclass": "Branched-chain amino acids (BCAA)",
        "pathway": "BCAA metabolism",
        "function": "Essential amino acid, protein synthesis",
        "ripening_role": "Precursor to isoamyl esters (banana aroma compounds)"
    },
    "GLN": {
        "common_name": "Glutamine",
        "chemical_class": "Amino acids",
        "subclass": "Amide amino acids",
        "pathway": "Nitrogen metabolism",
        "function": "Nitrogen donor, protein synthesis",
        "ripening_role": "Nitrogen remobilization during senescence"
    },
    "GLU": {
        "common_name": "Glutamic acid (Glutamate)",
        "chemical_class": "Amino acids",
        "subclass": "Acidic amino acids",
        "pathway": "Central nitrogen metabolism",
        "function": "Neurotransmitter, umami taste",
        "ripening_role": "Nitrogen metabolism hub, GABA precursor"
    },
    "GLY": {
        "common_name": "Glycine",
        "chemical_class": "Amino acids",
        "subclass": "Simple amino acids",
        "pathway": "One-carbon metabolism",
        "function": "Protein synthesis, precursor to many compounds",
        "ripening_role": "General metabolic activity marker"
    },
    "PHE": {
        "common_name": "Phenylalanine",
        "chemical_class": "Amino acids",
        "subclass": "Aromatic amino acids",
        "pathway": "Shikimate pathway → phenylpropanoid pathway",
        "function": "Precursor to phenolic compounds",
        "ripening_role": "Precursor to aroma compounds and phenolics"
    },
    "LEU": {
        "common_name": "Leucine",
        "chemical_class": "Amino acids",
        "subclass": "Branched-chain amino acids (BCAA)",
        "pathway": "BCAA metabolism",
        "function": "Essential amino acid",
        "ripening_role": "Direct precursor to isoamyl acetate (THE banana smell)"
    },
    "TYR": {
        "common_name": "Tyrosine",
        "chemical_class": "Amino acids",
        "subclass": "Aromatic amino acids",
        "pathway": "Shikimate pathway → tyrosine pathway",
        "function": "Precursor to dopamine, melanin",
        "ripening_role": "Browning reactions (polyphenol oxidase substrates)"
    },
    "ILE": {
        "common_name": "Isoleucine",
        "chemical_class": "Amino acids",
        "subclass": "Branched-chain amino acids (BCAA)",
        "pathway": "BCAA metabolism",
        "function": "Essential amino acid",
        "ripening_role": "Precursor to 2-methylbutyl esters (banana aroma)"
    },
    
    # ============================================================================
    # METABOLITES (from LC-MS, central carbon metabolism)
    # ============================================================================
    "47_SHIK 0.0 : Conc. (nM)": {
        "common_name": "Shikimic acid (Shikimate)",
        "chemical_class": "Organic acids",
        "subclass": "Cyclohexene carboxylic acids",
        "pathway": "Shikimate pathway (aromatic amino acid biosynthesis)",
        "function": "Precursor to Phe, Tyr, Trp (aromatic amino acids)",
        "ripening_role": "Increases - pathway active for aroma compound synthesis"
    },
    "48_SUCC_0.0 : Conc. (nM)": {
        "common_name": "Succinic acid (Succinate)",
        "chemical_class": "Organic acids",
        "subclass": "Dicarboxylic acids",
        "pathway": "TCA cycle (Krebs cycle)",
        "function": "Central energy metabolism intermediate",
        "ripening_role": "Increases - elevated respiration during climacteric ripening"
    },
    "21_FUM_0.0 : Conc. (nM)": {
        "common_name": "Fumaric acid (Fumarate)",
        "chemical_class": "Organic acids",
        "subclass": "Dicarboxylic acids",
        "pathway": "TCA cycle",
        "function": "Energy metabolism, succinate → fumarate → malate",
        "ripening_role": "Increases with respiration during ripening"
    },
    "11_AMP 0.0 : Conc. (nM)": {
        "common_name": "Adenosine monophosphate (AMP)",
        "chemical_class": "Nucleotides",
        "subclass": "Purine ribonucleotides",
        "pathway": "Purine nucleotide metabolism",
        "function": "Energy signaling, RNA building block",
        "ripening_role": "ATP degradation product - energy turnover during senescence"
    },
    "53_UMP 0.0 : Conc. (nM)": {
        "common_name": "Uridine monophosphate (UMP)",
        "chemical_class": "Nucleotides",
        "subclass": "Pyrimidine ribonucleotides",
        "pathway": "Pyrimidine nucleotide metabolism",
        "function": "RNA synthesis, carbohydrate metabolism",
        "ripening_role": "Nucleotide turnover during cell wall breakdown"
    },
    "DHAP_0.0 : Conc. (nM)": {
        "common_name": "Dihydroxyacetone phosphate (DHAP)",
        "chemical_class": "Carbohydrate phosphates",
        "subclass": "Triose phosphates",
        "pathway": "Glycolysis / Calvin cycle",
        "function": "Central carbohydrate metabolism intermediate",
        "ripening_role": "Starch → sugar conversion, increased glycolysis"
    },
    "44_R5P_0.0 : Conc. (nM)": {
        "common_name": "Ribose-5-phosphate (R5P)",
        "chemical_class": "Carbohydrate phosphates",
        "subclass": "Pentose phosphates",
        "pathway": "Pentose phosphate pathway",
        "function": "NADPH production, nucleotide/aromatic synthesis",
        "ripening_role": "Provides precursors for aromatic compound synthesis"
    },
    "33_LAC_0.0 : Conc. (nM)": {
        "common_name": "Lactic acid (Lactate)",
        "chemical_class": "Organic acids",
        "subclass": "Hydroxy acids",
        "pathway": "Glycolysis (fermentation)",
        "function": "Anaerobic metabolism end-product",
        "ripening_role": "Indicates some fermentative metabolism during ripening"
    },
    "28_GMP 0.0 : Conc. (nM)": {
        "common_name": "Guanosine monophosphate (GMP)",
        "chemical_class": "Nucleotides",
        "subclass": "Purine ribonucleotides",
        "pathway": "Purine nucleotide metabolism",
        "function": "RNA building block, signaling",
        "ripening_role": "Nucleotide degradation during senescence"
    },
    "10_ADP 0.0 : Conc. (nM)": {
        "common_name": "Adenosine diphosphate (ADP)",
        "chemical_class": "Nucleotides",
        "subclass": "Purine ribonucleotides",
        "pathway": "Energy metabolism (ATP cycle)",
        "function": "Energy currency, ADP ↔ ATP",
        "ripening_role": "High energy turnover during climacteric respiration"
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_compound_info(feature_name):
    """Get info for a compound by its feature name."""
    return COMPOUND_DATABASE.get(feature_name, None)


def classify_compounds(feature_list):
    """
    Classify a list of features into chemical classes.
    
    Returns dict with feature names as keys and classification info as values.
    """
    results = {}
    for feature in feature_list:
        info = get_compound_info(feature)
        if info:
            results[feature] = info
        else:
            print(f"⚠️ Warning: No classification for '{feature}'")
    return results


def summarize_by_class(feature_list, class_key='chemical_class'):
    """
    Count compounds by chemical class.
    
    Parameters
    ----------
    class_key : str
        'chemical_class' or 'pathway' or 'subclass'
    """
    from collections import Counter
    
    classifications = classify_compounds(feature_list)
    classes = [info[class_key] for info in classifications.values()]
    
    return Counter(classes)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_chemical_classes(feature_list, title="Chemical Classes", 
                          class_key='chemical_class', top_n=10):
    """
    Plot distribution of chemical classes for your features.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    
    # Get classifications
    class_counts = summarize_by_class(feature_list, class_key=class_key)
    
    # Convert to DataFrame
    import pandas as pd
    df_plot = pd.DataFrame(class_counts.items(), columns=['Class', 'Count'])
    df_plot = df_plot.sort_values('Count', ascending=False).head(top_n)
    
    # Color mapping
    color_map = {
        'Esters': '#f39c12',
        'Alcohols': '#1abc9c',
        'Amino acids': '#2ecc71',
        'Organic acids': '#e74c3c',
        'Nucleotides': '#3498db',
        'Carbohydrate phosphates': '#e67e22',
        'Ketones': '#9b59b6',
        'Isothiocyanates': '#34495e',
    }
    
    colors = [color_map.get(c, '#95a5a6') for c in df_plot['Class']]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_plot['Class'], df_plot['Count'], color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, df_plot['Count'])):
        ax.text(count + 0.1, i, str(count), va='center', fontweight='bold')
    
    ax.set_xlabel('Number of Key Biomarkers', fontsize=13, fontweight='bold')
    ax.set_ylabel('Chemical Class', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, df_plot


def create_summary_table(feature_list):
    """
    Create a detailed summary table of your compounds.
    """
    import pandas as pd
    
    classifications = classify_compounds(feature_list)
    
    rows = []
    for feature, info in classifications.items():
        # Clean feature name for display
        display_name = feature.replace(' Area', '').replace(' : Conc. (nM)', '').strip()
        
        rows.append({
            'Feature': display_name,
            'Common Name': info.get('common_name', ''),
            'Chemical Class': info.get('chemical_class', ''),
            'Pathway': info.get('pathway', ''),
            'Ripening Role': info.get('ripening_role', ''),
        })
    
    return pd.DataFrame(rows)