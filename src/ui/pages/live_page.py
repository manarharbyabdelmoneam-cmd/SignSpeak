"""
UI Page: Live Translation — FUTURE FEATURE, NOT IN MVP
=========================================================
Thin wrapper page for the future live-camera translation feature.
Delegates entirely to camera_component (currently a placeholder).

Kept as a real page (not deleted) so the sidebar navigation and overall
app structure match the original architecture, and so this feature can
be built out later without touching app.py's routing.
"""

import streamlit as st

from src.ui.components.camera_component import render_camera_component


def render_live_page() -> None:
    """Renders the live-translation page (placeholder for the MVP)."""
    st.markdown("## 📷 الترجمة المباشرة")
    render_camera_component()
