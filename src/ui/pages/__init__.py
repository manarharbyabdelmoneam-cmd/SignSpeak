"""
UI Pages - top-level Streamlit views, composed from ui/components/.

Each page is a single render_*_page() function that app.py wires up to
the sidebar/navigation. Pages only talk to services/ (never directly to
core/ or MediaPipe/Keras).
"""

from .home_page import render_home_page
from .upload_page import render_upload_page
from .live_page import render_live_page
from .dashboard_page import render_dashboard_page

__all__ = [
    "render_home_page",
    "render_upload_page",
    "render_live_page",
    "render_dashboard_page",
]
