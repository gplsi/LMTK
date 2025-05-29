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

    .. attribute:: NONE
       :no-index:
    .. attribute:: ERRORS
       :no-index:
    .. attribute:: WARNINGS
       :no-index:
    .. attribute:: INFO
       :no-index:
    .. attribute:: DEBUG
       :no-index:
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

    .. attribute:: COLORS
       :no-index:
    .. attribute:: RESET
       :no-index:
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
        level (:data:`VerboseLevel`): The desired verbosity level. Defaults to :data:`VerboseLevel.INFO` :no-index:.
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
    
    # Get the numeric level from the mapping
    numeric_level = level_mapping.get(level, logging.INFO)
    
    # Set the logger's level
    logger.setLevel(numeric_level)
    
    # If level is NONE, disable the logger
    if level == VerboseLevel.NONE:
        logger.disabled = True
    else:
        logger.disabled = False
        
    return logger


def setup_logging(level: VerboseLevel = VerboseLevel.INFO) -> None:
    """
    Set up logging for the entire application.
    
    This function configures the root logger with the specified verbosity level
    and adds a console handler with the custom formatter.
    
    Args:
        level (VerboseLevel): The desired verbosity level. Defaults to VerboseLevel.INFO.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:  
        root_logger.removeHandler(handler)
    
    # Set the root logger level
    numeric_level = level_mapping.get(level, logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter and add it to the handler
    formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the root logger
    root_logger.addHandler(console_handler)
    
    # If level is NONE, disable the logger
    if level == VerboseLevel.NONE:
        root_logger.disabled = True