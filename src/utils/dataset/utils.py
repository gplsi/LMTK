import os
from typing import Dict


SUPPORTED_EXTENSIONS = ["txt", "csv", "json"]


def scan_directory(path, extension: str = None) -> Dict:
    """
    Scans the given directory for text files and returns a dictionary of data sources and their files.

    Args:
        path (str): Path to the directory containing subfolders with text files.

    Returns:
        dict: A dictionary where keys are data sources (folder names) and values are lists of text file paths.
    """
    if (extension is not None) and (extension not in SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported file extension: {extension}.")

    data_sources = {}
    for root, dirs, files in os.walk(path):
        source = os.path.basename(root)
        if extension is not None:
            data_files = [
                os.path.join(root, file)
                for file in files
                if file.endswith(f".{extension}")
            ]  # Filter by extension
        else:
            data_files = [
                os.path.join(root, file)
                for file in files
                if file.split(".")[-1] in SUPPORTED_EXTENSIONS
            ]

        if data_files:
            data_sources[source] = data_files  # Exclude the root directory itself

    return data_sources
