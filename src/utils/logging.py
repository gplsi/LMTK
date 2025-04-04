"""
This module provides utility functions to configure and manage logging for an application.
It defines custom log levels, a custom log formatter with color codes, and helper functions to
get and set logger verbosity levels. The implementation leverages the standard Python logging library.
"""

import logging
from enum import IntEnum
from typing import Optional
import sys

class VerboseLevel(IntEnum):
    """
    Enumeration for specifying verbosity levels.

    Attributes:
        NONE (int): No logging output.
        ERRORS (int): Only error messages.
        WARNINGS (int): Warning messages.
        INFO (int): Informational messages.
        DEBUG (int): Debugging messages.
    """
    
    NONE = 0
    ERRORS = 1
    WARNINGS = 2
    INFO = 3
    DEBUG = 4
    
level_mapping = {
        VerboseLevel.NONE: logging.CRITICAL + 1,
        VerboseLevel.ERRORS: logging.ERROR,
        VerboseLevel.WARNINGS: logging.WARNING,
        VerboseLevel.INFO: logging.INFO,
        VerboseLevel.DEBUG: logging.DEBUG
    }

class CustomFormatter(logging.Formatter):
    """
    Custom log formatter that adds ANSI color codes to log messages based on their log level.

    This formatter enhances the readability of logs when printed to a terminal that supports ANSI colors.
    
    Attributes:
        COLORS (dict): Maps logging levels to ANSI escape sequences for color formatting.
        RESET (str): ANSI escape sequence to reset color formatting.
    """
    
    COLORS = {
        logging.DEBUG: '\033[0;36m',    # Cyan
        logging.INFO: '\033[0;32m',     # Green
        logging.WARNING: '\033[0;33m',  # Yellow
        logging.ERROR: '\033[0;31m',    # Red
        logging.CRITICAL: '\033[0;35m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color and avoid double formatting"""
        color = self.COLORS.get(record.levelno)
        record.levelname = f'{color}{record.levelname}{self.RESET}'
        record.msg = f'{color}{record.msg}{self.RESET}'
        
        return super().format(record)


def get_logger(
    name: str,
    level: Optional[VerboseLevel] = VerboseLevel.INFO
) -> logging.Logger:
    """
    Creates and configures a logger with a custom formatter and specified verbosity level.
    
    Args:
        name (str): The name of the logger (typically __name__).
        level (Optional[VerboseLevel]): The desired verbosity level. Defaults to VerboseLevel.INFO.
        rank (Optional[int]): The process rank in distributed training. Used to filter logs.
    
    Returns:
        logging.Logger: A logger instance configured with the specified settings.
    """
    
    if (level < VerboseLevel.NONE) or (level > VerboseLevel.DEBUG):
        raise ValueError("Invalid verbose level. Must be between None and Debug.")
    
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger.addHandler(console_handler)
    
    # Map VerboseLevel to logging levels
    level_mapping = {
        VerboseLevel.NONE: logging.CRITICAL + 1,
        VerboseLevel.ERRORS: logging.ERROR,
        VerboseLevel.WARNINGS: logging.WARNING,
        VerboseLevel.INFO: logging.INFO,
        VerboseLevel.DEBUG: logging.DEBUG
    }
    
    logger.setLevel(level_mapping.get(level, logging.INFO))
    
    # Disable logging completely if NONE
    if level == VerboseLevel.NONE:
        logger.disabled = True
    
    return logger

def set_logger_level(logger: logging.Logger, level: VerboseLevel) -> logging.Logger:
    """
    Updates the verbosity level of an existing logger instance.

    This function modifies the logger's level according to the provided VerboseLevel. If the level is NONE,
    it also disables the logger to prevent any log output.

    Args:
        logger (logging.Logger): The logger whose level is to be updated.
        level (VerboseLevel): The new verbosity level.

    Returns:
        logging.Logger: The logger with the updated logging level.
    """
    
    logger.setLevel(level_mapping.get(level, logging.INFO))
    
    # Disable logging completely if NONE
    if level == VerboseLevel.NONE:
        logger.disabled = True
        
    return logger