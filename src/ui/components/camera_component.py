"""
UI Component: Camera (Live Translation) — FUTURE FEATURE, NOT IN MVP
=======================================================================
Placeholder for real-time webcam-based sign translation.

We agreed the MVP ships upload-only: live camera needs streamlit-webrtc
(WebRTC + TURN servers), which is fragile on Streamlit Cloud's free tier
and adds real deployment risk. This component exists only so the
architecture stays consistent with the original design and so live
translation can be built later without restructuring anything.

NOT imported by any page in the current app.
"""

from typing import Optional

import streamlit as st


def render_camera_component() -> Optional[dict]:
    """
    Placeholder entry point for the future live-camera page.

    Currently just informs the user the feature isn't available yet,
    instead of rendering a broken/half-built camera widget. Returns None
    always (no result to hand back).

    Future implementation sketch (not built now):
        1. Use `streamlit-webrtc` to get a live video frame stream.
        2. Per frame: LandmarkExtractor.extract_frame_features(frame)
        3. Feed each 153-feature vector into BufferService.add_frame()
        4. When BufferService returns a non-None result, render it via
           render_result_component() and call buffer.reset()
    """
    st.markdown("### الترجمة المباشرة بالكاميرا")
    st.info(
        "الترجمة المباشرة بالكاميرا لسه مش متاحة في النسخة دي من التطبيق 🚧\n\n"
        "استخدمي تبويب **رفع فيديو** بدلها."
    )
    return None
