"""
SignBridge AI - Streamlit Entry Point
========================================
Wires together config, logging, cached services, and page routing.
This is the only file Streamlit actually runs (`streamlit run app.py`).

Responsibilities here ONLY:
    - App-wide setup (logging, page config)
    - Creating expensive resources ONCE via st.cache_resource
      (MediaPipe + TFLite model load is slow - must not repeat per rerun)
    - Sidebar navigation / routing to the right page

No ML logic, no UI layout details - those live in services/ and ui/.
"""

import streamlit as st

from config.settings import APP_TITLE, APP_ICON, PAGE_LAYOUT
from src.utils.logger import setup_logging
from src.services.model_service import ModelService
from src.services.video_service import VideoService
from src.ui.pages import (
    render_home_page,
    render_upload_page,
    render_live_page,
    render_dashboard_page,
)

# ------------------------------------------------------------
# App-wide setup (runs once per process, cheap)
# ------------------------------------------------------------
setup_logging()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
)


# ------------------------------------------------------------
# Cached, expensive resources - created ONCE, reused across reruns
# and across all users on the same server instance.
# ------------------------------------------------------------
@st.cache_resource(show_spinner="بنجهز الموديل... أول مرة بتاخد شوية وقت")
def get_model_service() -> ModelService:
    return ModelService()


def get_video_service() -> VideoService:
    # Cheap to construct (just wraps model_service), no need to cache
    # separately - but model_service itself IS cached above.
    return VideoService(model_service=get_model_service())


# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
PAGES = {
    "🏠 الرئيسية": render_home_page,
    "📤 رفع فيديو": render_upload_page,
    "📷 الترجمة المباشرة": render_live_page,
    "📊 الإحصائيات": render_dashboard_page,
}

st.sidebar.markdown(f"## {APP_ICON} {APP_TITLE}")
st.sidebar.divider()

selected_page = st.sidebar.radio(
    label="التنقل",
    options=list(PAGES.keys()),
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("مشروع تخرج - برنامج Digilians")

# ------------------------------------------------------------
# Route to the selected page
# ------------------------------------------------------------
page_function = PAGES[selected_page]

if selected_page == "📤 رفع فيديو":
    page_function(get_video_service())
else:
    page_function()
