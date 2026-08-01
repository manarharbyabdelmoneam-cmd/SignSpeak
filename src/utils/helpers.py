"""
Utils: Helpers
================
Small, generic, framework-agnostic utility functions used across the
project. Nothing here depends on Streamlit, MediaPipe, or Keras -- if a
helper needs those, it belongs in services/ or ui/, not here.
"""

from datetime import datetime
from pathlib import Path


def format_confidence(confidence: float) -> str:
    """0.847 -> '84.7%'"""
    return f"{confidence * 100:.1f}%"


def format_file_size(size_bytes: int) -> str:
    """1_500_000 -> '1.4 MB'"""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_duration(seconds: float) -> str:
    """125.4 -> '2:05'  |  8.2 -> '0:08'"""
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes}:{secs:02d}"


def format_timestamp(dt: datetime | None = None, arabic: bool = True) -> str:
    """
    Returns a display-friendly timestamp, e.g. '2026-08-01 14:32'.
    `arabic` is currently unused but kept for future locale formatting
    (e.g. Arabic month names) without changing call sites later.
    """
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M")


def safe_filename(filename: str) -> str:
    """
    Strips characters that could cause issues in temp/output paths while
    keeping the file readable. Not a full security sanitizer - only for
    display/logging purposes (actual uploads go through tempfile with a
    random name, see video_service.py).
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    cleaned_stem = "".join(
        char for char in stem if char.isalnum() or char in (" ", "-", "_")
    ).strip()
    return f"{cleaned_stem or 'video'}{suffix}"


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Clamp a value into [min_value, max_value], e.g. for progress bars."""
    return max(min_value, min(value, max_value))


def get_confidence_level_label(confidence: float, threshold: float) -> str:
    """
    Maps a raw confidence score to a coarse Arabic label for display,
    e.g. in the dashboard or result card.
    """
    if confidence >= max(threshold, 0.85):
        return "ثقة عالية"
    if confidence >= threshold:
        return "ثقة متوسطة"
    return "ثقة منخفضة"
