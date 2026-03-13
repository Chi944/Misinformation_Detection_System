import logging
import os
import sys


_loggers = {}


def get_logger(name, level=None, log_file=None):
    """
    Get or create a named logger with consistent formatting.

    Returns a cached logger if already created with the same name.
    Logs to stdout by default, optionally also to a file.

    Args:
        name (str): logger name (typically __name__)
        level (int, optional): logging level. Defaults to INFO or
                               DEBUG if LOG_LEVEL=DEBUG env var set
        log_file (str, optional): path to write log file
    Returns:
        logging.Logger: configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    if level is None:
        env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    logger.propagate = False
    _loggers[name] = logger
    return logger
