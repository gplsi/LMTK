import os
from typing import Dict
from src.utils.logging import get_logger

SUPPORTED_EXTENSIONS = ["txt", "csv", "json", "jsonl"]

# Create a logger for this module
local_logger = get_logger(__name__)

def scan_directory(path, extension: str = None, logger = local_logger) -> Dict:
    """
    Scans the given directory for text files and returns a dictionary of data sources and their files.

    Args:
        path (str): Path to the directory containing subfolders with text files.
        extension (str, optional): If provided, only files with this extension will be included.
        logger (str, optional): If provided, only files with this extension will be included.

    Returns:
        dict: A dictionary where keys are data sources (folder names) and values are lists of text file paths.
    """
    if (extension is not None) and (extension not in SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported file extension: {extension}.")

    logger.info(f"Starting directory scan in: {path}")
    total_files_found = 0
    total_dirs_scanned = 0
    data_sources = {}
    
    for root, dirs, files in os.walk(path):
        total_dirs_scanned += 1
        source = os.path.basename(root)
        
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
