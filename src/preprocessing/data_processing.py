import pandas as pd

def load_and_impute(file_path, group_col, fill_value=None, impute=True):
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
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    df = pd.read_csv(file_path)
    
    if impute:
        grouped = df.groupby(group_col)
        def impute_missing(group):
            group = group.copy()
            for col in group.columns:
                if col != group_col and group[col].isnull().any():
                    group[col] = group[col].fillna(group[col].median())
            return group
        df = grouped.apply(impute_missing)
    
    if fill_value is not None:
        df = df.fillna(fill_value)
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)
    return df
