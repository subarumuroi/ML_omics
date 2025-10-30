"""
Input/output utilities for saving and loading results.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def save_results(results_dict, filepath, verbose=True):
    """
    Save results dictionary to file.
    
    Supports JSON and pickle formats. JSON is more portable but cannot
    handle all Python objects. Pickle can handle any Python object but
    is Python-specific.
    
    Parameters
    ----------
    results_dict : dict
        Results to save
    filepath : str or Path
        Output filepath (supports .json, .pkl, .pickle)
    verbose : bool, default=True
        Print confirmation message
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.json':
        # Convert numpy arrays to lists for JSON serialization
        serializable = _make_json_serializable(results_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    elif filepath.suffix in ['.pkl', '.pickle']:
        with open(filepath, 'wb') as f:
            pickle.dump(results_dict, f)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .json, .pkl, or .pickle")
    
    if verbose:
        print(f"Results saved to: {filepath}")


def load_results(filepath, verbose=True):
    """
    Load results from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to results file
    verbose : bool, default=True
        Print confirmation message
        
    Returns
    -------
    dict
        Loaded results
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    
    elif filepath.suffix in ['.pkl', '.pickle']:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    if verbose:
        print(f"Results loaded from: {filepath}")
    
    return results


def save_dataframe(df, filepath, verbose=True, **kwargs):
    """
    Save DataFrame to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or Path
        Output filepath
    verbose : bool, default=True
        Print confirmation
    **kwargs : dict
        Additional arguments passed to pd.DataFrame.to_csv
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False, **kwargs)
    
    if verbose:
        print(f"DataFrame saved to: {filepath}")


def save_figure(fig, filepath, dpi=300, bbox_inches='tight', verbose=True):
    """
    Save matplotlib figure.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str or Path
        Output filepath
    dpi : int, default=300
        Resolution in dots per inch
    bbox_inches : str, default='tight'
        Bounding box specification
    verbose : bool, default=True
        Print confirmation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    
    if verbose:
        print(f"Figure saved to: {filepath}")


def create_results_directory(base_path='results', run_name=None):
    """
    Create directory structure for saving results.
    
    Parameters
    ----------
    base_path : str or Path, default='results'
        Base directory for results
    run_name : str, optional
        Name for this run (timestamp will be added if None)
        
    Returns
    -------
    dict
        Dictionary with paths to subdirectories
    """
    from datetime import datetime
    
    if run_name is None:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    base = Path(base_path) / run_name
    
    dirs = {
        'base': base,
        'figures': base / 'figures',
        'data': base / 'data',
        'models': base / 'models',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created results directory: {base}")
    
    return dirs


def _make_json_serializable(obj):
    """
    Convert object to JSON-serializable format.
    
    Handles numpy arrays, numpy scalars, and nested structures.
    """
    if isinstance(obj, dict):
        return {key: _make_json_serializable(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def save_model(model, filepath, verbose=True):
    """
    Save trained model to file.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model
    filepath : str or Path
        Output filepath (.pkl or .pickle)
    verbose : bool, default=True
        Print confirmation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix not in ['.pkl', '.pickle']:
        filepath = filepath.with_suffix('.pkl')
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    if verbose:
        print(f"Model saved to: {filepath}")


def load_model(filepath, verbose=True):
    """
    Load trained model from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to model file
    verbose : bool, default=True
        Print confirmation
        
    Returns
    -------
    sklearn estimator
        Loaded model
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    if verbose:
        print(f"Model loaded from: {filepath}")
    
    return model