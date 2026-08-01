"""
Utils - small, generic, framework-agnostic helper functions and logging
setup. No dependencies on services/, ui/, or core/.
"""

from .logger import setup_logging
from .helpers import (
    format_confidence,
    format_file_size,
    format_duration,
    format_timestamp,
    safe_filename,
    clamp,
    get_confidence_level_label,
)

__all__ = [
    "setup_logging",
    "format_confidence",
    "format_file_size",
    "format_duration",
    "format_timestamp",
    "safe_filename",
    "clamp",
    "get_confidence_level_label",
]
