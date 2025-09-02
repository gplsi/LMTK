"""
This module provides a utility to recursively scan directories for files with specific or 
supported file extensions. The primary function, scan_directory, traverses a given directory 
tree and groups files by their parent folder, filtering them by either a provided extension 
or a set of default supported extensions.

Supported file extensions include: "txt", "csv", "json", "jsonl", and "parquet".
"""

import os
from typing import Dict
from src.utils.logging import get_logger

SUPPORTED_EXTENSIONS = ["txt", "csv", "json", "jsonl", "parquet"]

# Create a logger for this module
local_logger = get_logger(__name__)


def scan_directory(path, extension: str = None, logger = local_logger) -> Dict:
    """
    Scan the provided directory (and its subdirectories) for files matching a given extension
    or the default set of supported extensions.

    The function traverses the entire directory tree starting from 'path'. For each directory,
    it collects files that satisfy the extension filter:

      - If an extension is explicitly provided, only files ending with that extension will be collected.
      - If no extension is provided, any file with an extension listed in SUPPORTED_EXTENSIONS is collected.

    The results are returned in a dictionary where the keys are the names of the directories (data sources)
    and the values are lists containing the full paths to the matching files.

    Args:
        path (str): Path to the directory containing subfolders with text files.
        extension (str, optional): If provided, only files with this extension will be included.
        logger (Logger, optional): Logger instance to use for logging messages.

    Returns:
        Dict[str, list]: A dictionary mapping each data source (directory name) to a list of file paths
                         that match the specified filter criteria.

    Raises:
        ValueError: If an extension is provided that is not among the supported extensions.

    """
    
    if (extension is not None) and (extension not in SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported file extension: {extension}.")

    logger.info(f"Starting directory scan in: {path}")
    total_files_found = 0
    total_dirs_scanned = 0
    data_sources = {}
    
    for root, dirs, files in os.walk(path):
        total_dirs_scanned += 1
        source = os.path.relpath(root, path)  # Use relative path from base path to avoid key collisions
        
        if extension is not None:
            data_files = [
                os.path.join(root, file)
                for file in files
                if file.endswith(f".{extension}")
            ]  # Filter by extension
            matching_count = len(data_files)
            total_count = len(files)
            logger.debug(f"Directory: {root} - Found {matching_count}/{total_count} files with .{extension} extension")
        else:
            data_files = [
                os.path.join(root, file)
                for file in files
                if file.split(".")[-1] in SUPPORTED_EXTENSIONS
            ]
            matching_count = len(data_files)
            total_count = len(files)
            logger.debug(f"Directory: {root} - Found {matching_count}/{total_count} files with supported extensions {SUPPORTED_EXTENSIONS}")

        if data_files:
            data_sources[source] = data_files
            total_files_found += len(data_files)

    logger.debug(f"Scan completed: Found {total_files_found} matching files across {total_dirs_scanned} directories")
    if not data_sources:
        logger.warning(f"No matching files found in {path}")
        
    return data_sources
