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
        """
        Format the specified log record as text, inserting ANSI color codes.

        This overridden method wraps the log level name and the log message with color codes
        based on the record's log level.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with ANSI color codes.
        """
        # Retrieve the color associated with the record's level; default is None if not found.
        color = self.COLORS.get(record.levelno)
        # Decorate the level name and the message with the retrieved color and RESET code.
        record.levelname = f'{color}{record.levelname}{self.RESET}'
        record.msg = f'{color}{record.msg}{self.RESET}'
        # Delegate the rest of the formatting to the parent class.
        return super().format(record)


def get_logger(
    name: str,
    level: Optional[VerboseLevel] = VerboseLevel.INFO
) -> logging.Logger:
    """
    Creates and configures a logger with a custom formatter and specified verbosity level.

    This function sets up a logger that outputs to the console (stdout) using the custom colorized
    formatter. It clears any pre-existing handlers to avoid duplicate log entries and maps the provided
    verbosity level (from the VerboseLevel enum) to a corresponding standard logging level.

    Args:
        name (str): The name of the logger (typically __name__).
        level (Optional[VerboseLevel]): The desired verbosity level. Defaults to VerboseLevel.INFO.

    Raises:
        ValueError: If the provided verbosity level is not within the acceptable range.

    Returns:
        logging.Logger: A logger instance configured with the specified settings.
    """
    # Validate that the provided level is within the valid range.
    if (level < VerboseLevel.NONE) or (level > VerboseLevel.DEBUG):
        raise ValueError("Invalid verbose level. Must be between None and Debug.")
    
    # Retrieve (or create) a logger instance with the given name.
    logger = logging.getLogger(name)
    
    # Remove any existing handlers to prevent duplicate log outputs.
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create a stream handler that sends log messages to stdout.
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set a custom formatter for the handler to include colored logs and a specific message format.
    console_handler.setFormatter(CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Attach the handler to the logger.
    logger.addHandler(console_handler)
    
    # Local mapping for the current logger instantiation to map VerboseLevel to standard logging levels.
    level_mapping = {
        VerboseLevel.NONE: logging.CRITICAL + 1,
        VerboseLevel.ERRORS: logging.ERROR,
        VerboseLevel.WARNINGS: logging.WARNING,
        VerboseLevel.INFO: logging.INFO,
        VerboseLevel.DEBUG: logging.DEBUG
    }
    
    # Set the logger's level based on the provided verbosity level.
    logger.setLevel(level_mapping.get(level, logging.INFO))
    
    # If verbosity is set to NONE, disable the logger entirely.
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
    # Update the logger's level using the global mapping.
    logger.setLevel(level_mapping.get(level, logging.INFO))
    
    # Disable the logger if the verbosity level is NONE.
    if level == VerboseLevel.NONE:
        logger.disabled = True
        
    return logger
