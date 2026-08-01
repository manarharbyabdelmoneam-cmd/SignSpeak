"""
UI Components - reusable Streamlit building blocks.

Note: camera_component is part of the live-translation feature, which is
NOT wired into the MVP (upload-only). It's included here for structural
completeness but isn't imported by any page yet.
"""

from .upload_component import render_upload_component
from .result_component import render_result_component
from .camera_component import render_camera_component

__all__ = [
    "render_upload_component",
    "render_result_component",
    "render_camera_component",
]
