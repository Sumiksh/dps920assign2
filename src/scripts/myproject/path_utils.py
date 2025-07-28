# src/my_project/path_utils.py

import os

def find_project_root(marker_dirs=('data',), max_up=5):
    """
    Starting from this file’s directory, climb up until you find
    a folder that contains *all* of the names in `marker_dirs`.
    """
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(max_up):
        # Check if all marker_dirs exist here
        if all(os.path.isdir(os.path.join(current, m)) for m in marker_dirs):
            return current
        # Otherwise, go up one level
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        f"Could not find project root containing {marker_dirs} within {max_up} levels above {__file__}"
    )

def data_path(*subpaths):
    """
    Returns an absolute path under the top-level `data/` directory,
    even if you call this from src/scripts, tests/, notebooks/, etc.
    
    Example:
        data_csv = data_path("Q1", "house_price.csv")
        cascade = data_path("Q2", "haarcascade_frontalcatface.xml")
    """
    project_root = find_project_root(marker_dirs=('data',))
    return os.path.join(project_root, "data", *subpaths)
