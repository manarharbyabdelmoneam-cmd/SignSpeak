"""
Utils: Logger
===============
Centralized logging setup. Every module in the project uses
`logging.getLogger(__name__)` directly (see model_service.py,
video_service.py, etc.) - this file just configures the root logger
once, at app startup, so those loggers actually write somewhere useful.
"""

import logging
import sys

from config.settings import LOG_FILE_PATH, LOG_LEVEL

_configured = False


def setup_logging() -> None:
    """
    Configure the root logger with both a file handler and a console
    handler. Safe to call multiple times (e.g. across Streamlit reruns);
    only configures once per process.

    Call this once, early, in app.py before anything else runs.
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Quiet down noisy third-party libraries so signbridge logs aren't
    # drowned out.
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("mediapipe").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info("Logging configured (level=%s)", LOG_LEVEL)
