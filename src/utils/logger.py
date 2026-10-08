import logging


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a module logger."""
    return logging.getLogger(name)