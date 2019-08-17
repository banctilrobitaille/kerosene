import os
from glob import glob
from typing import List, Tuple


def split_filename(filepath: str) -> Tuple[str, str, str]:
    """
    Split a filepath into the directory, base, and extension

    Args:
        filepath (str): The base file path.

    Returns:
        Tuple: The complete file path, base path and file extension.
    """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def extract_file_paths(path: str, ext='*.nii*') -> List[str]:
    """
    Grab all `ext` files in a directory and sort them for consistency.

    Args:
        path (str): File path.
        ext (str): File's extension to grab.

    Returns:
        list: A list of string containing every file paths.
    """
    file_paths = sorted(glob(os.path.join(path, ext)))
    return file_paths
