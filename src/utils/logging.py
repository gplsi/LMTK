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


# Global mapping that aligns our custom VerboseLevel with standard logging levels.
# Note: For the NONE level, logging.CRITICAL + 1 is used to effectively disable logging.
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
        logging.DEBUG: '\033[0;36m',    # Cyan for DEBUG level
        logging.INFO: '\033[0;32m',     # Green for INFO level
        logging.WARNING: '\033[0;33m',  # Yellow for WARNING level
        logging.ERROR: '\033[0;31m',    # Red for ERROR level
        logging.CRITICAL: '\033[0;35m'  # Magenta for CRITICAL level
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color and avoid double formatting"""
        if not hasattr(record, 'formatted_message'):
            # Only color and format the message once
            color = self.COLORS.get(record.levelno, '')
            record.levelname = f'{color}{record.levelname}{self.RESET}'
            record.msg = f'{color}{record.msg}{self.RESET}'
            record.formatted_message = True
            
            # Add process rank to log format if in distributed mode
            if hasattr(record, 'rank'):
                self._fmt = f'%(asctime)s [%(rank)s] - %(name)s - %(levelname)s - %(message)s'
            
        return super().format(record)


def get_logger(
    name: str,
    level: Optional[VerboseLevel] = VerboseLevel.INFO,
    rank: Optional[int] = None
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
    
    # Only configure handler if it hasn't been configured yet
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = CustomFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Set initial log level
    initial_level = level_mapping.get(level, logging.INFO)
    
    # Handle distributed training logging
    if rank is not None:
        if rank != 0:  # Non-main processes
            # Only show errors and warnings from non-main processes
            logger.setLevel(logging.WARNING)
            formatter = CustomFormatter(
                f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:  # Main process
            logger.setLevel(initial_level)
            formatter = CustomFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        # Update formatter for all handlers
        for handler in logger.handlers:
            handler.setFormatter(formatter)
    else:
        # No rank specified (non-distributed case)
        logger.setLevel(initial_level)
    
    if level == VerboseLevel.NONE:
        logger.disabled = True
    
    # Prevent log propagation to avoid duplicates
    logger.propagate = False
    
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
    # Update the logger's level using the global mapping.
    logger.setLevel(level_mapping.get(level, logging.INFO))
    
    # Disable the logger if the verbosity level is NONE.
    if level == VerboseLevel.NONE:
        logger.disabled = True
        
    return logger
