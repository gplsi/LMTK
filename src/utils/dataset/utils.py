"""
This module provides a utility to recursively scan directories for files with specific or 
supported file extensions. The primary function, scan_directory, traverses a given directory 
tree and groups files by their parent folder, filtering them by either a provided extension 
or a set of default supported extensions.

Supported file extensions include: "txt", "csv", and "json".
"""

import os
from typing import Dict, Optional

# List of file extensions that are supported by the scanning function.
SUPPORTED_EXTENSIONS = ["txt", "csv", "json"]


def scan_directory(path: str, extension: Optional[str] = None) -> Dict[str, list]:
    """
    Scan the provided directory (and its subdirectories) for files matching a given extension
    or the default set of supported extensions.

    The function traverses the entire directory tree starting from 'path'. For each directory,
    it collects files that satisfy the extension filter:
      - If an extension is explicitly provided, only files ending with that extension are collected.
      - If no extension is provided, any file with an extension listed in SUPPORTED_EXTENSIONS is collected.
    The results are returned in a dictionary where the keys are the names of the directories (data sources)
    and the values are lists containing the full paths to the matching files.

    Args:
        path (str): The root directory to start scanning from.
        extension (Optional[str]): The specific file extension to filter by (e.g., "txt", "csv", "json").
                                   If not None, must be one of the SUPPORTED_EXTENSIONS. Defaults to None.

    Returns:
        Dict[str, list]: A dictionary mapping each data source (directory name) to a list of file paths
                         that match the specified filter criteria.

    Raises:
        ValueError: If an extension is provided that is not among the supported extensions.
    """
    # Validate the provided file extension if it is not None.
    if (extension is not None) and (extension not in SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported file extension: {extension}.")

    # Dictionary to store the results: keys are directory names and values are lists of file paths.
    data_sources = {}

    # Recursively walk through the directory tree starting at 'path'.
    for root, dirs, files in os.walk(path):
        # Determine the name of the current directory (used as a key in the result dictionary).
        source = os.path.basename(root)

        # If an extension filter is provided, collect only files ending with that extension.
        if extension is not None:
            data_files = [
                os.path.join(root, file)
                for file in files
                if file.endswith(f".{extension}")
            ]
        else:
            # When no extension is provided, gather files whose extension is in SUPPORTED_EXTENSIONS.
            data_files = [
                os.path.join(root, file)
                for file in files
                if file.split(".")[-1] in SUPPORTED_EXTENSIONS
            ]

        # If matching files are found in the current directory, add them to the result dictionary.
        if data_files:
            data_sources[source] = data_files

    return data_sources
