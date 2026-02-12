"""
I/O utility functions.
"""

import os


def ensure_dir(path: str) -> str:
    """
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : str
        Directory path to create
        
    Returns
    -------
    str
        The same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path
