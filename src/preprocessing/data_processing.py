import numpy as np
import pandas as pd

def load_and_impute(file_path, group_col, fill_value=None, impute=True, drop_threshold=0.35):
    """
    Load a CSV file and optionally impute/fill missing values.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    group_col : str
        Column name to group by for median imputation.
    fill_value : numeric, optional
        If provided, fill any remaining missing values with this constant.
    impute : bool, default True
        If True, perform median imputation by group.
    drop_threshold : float, default 0.5
        Drop columns with more than this fraction of missing values (0.0 to 1.0).
        Set to None to keep all columns.
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    df = pd.read_csv(file_path, na_values=['', 'NA', 'na', 'NaN', 'nan'])
    
    # Drop columns with too many missing values (only numeric columns)
    if drop_threshold is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_fraction = df[numeric_cols].isna().sum() / len(df)
        cols_to_drop = missing_fraction[missing_fraction > drop_threshold].index.tolist()
        
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns with >{drop_threshold*100}% missing values")
        df = df.drop(columns=cols_to_drop)
    
    if impute:
        def impute_missing(group):
            group = group.copy()
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if group[col].isnull().any():
                    median_val = group[col].median()
                    # Only fill if median exists (not NaN from empty column)
                    if not pd.isna(median_val):
                        group[col] = group[col].fillna(median_val)
            return group
        
        # Don't use include_groups parameter - just let it include the group column
        df = df.groupby(group_col, group_keys=False).apply(impute_missing)
    
    if fill_value is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)
    return df