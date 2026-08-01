"""
UI Page: Upload
=================
Main MVP page: lets the user upload a video, runs it through
VideoService, and displays the result. Composes upload_component +
result_component - no ML logic lives here.
"""

import streamlit as st

from src.services.video_service import VideoService
from src.ui.components.upload_component import render_upload_component
from src.ui.components.result_component import render_result_component
from src.ui.pages.dashboard_page import log_prediction_to_session


def render_upload_page(video_service: VideoService) -> None:
    """
    Renders the upload page.

    `video_service` is created once (cached) in app.py and passed down,
    so MediaPipe/Keras aren't reloaded on every rerun/interaction.
    """
    st.markdown("## 📤 رفع فيديو")

    result = render_upload_component(video_service)

    if result is not None:
        st.divider()
        render_result_component(result)
        log_prediction_to_session(result)
