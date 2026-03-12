"""Logging utilities for the misinformation detector pipeline."""


def get_logger(name: str):
    """Return a standard library logger instance.

    Args:
        name: Logger name.

    Returns:
        logging.Logger: Configured logger.
    """

    import logging

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

