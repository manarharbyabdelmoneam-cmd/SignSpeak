"""
UI Component: Upload
======================
Renders the video upload widget and runs it through VideoService.

This component is intentionally "dumb" about ML details - it only knows
how to collect a file from the user, show a loading state, and hand back
either a prediction result dict or None (nothing uploaded / user hasn't
clicked run yet). All error handling is done by catching
VideoProcessingError, whose messages are already in Arabic and safe to
show directly.
"""

from typing import Optional

import streamlit as st

from src.services.video_service import VideoService, VideoProcessingError
from config.settings import MAX_UPLOAD_SIZE_MB

SUPPORTED_TYPES = ["mp4", "avi", "mov", "mkv"]


def render_upload_component(video_service: VideoService) -> Optional[dict]:
    """
    Renders the file uploader + "Translate" button.

    Returns:
        - A prediction result dict (same shape as ModelService.process_video())
          once processing succeeds.
        - None if nothing has been uploaded/processed yet.

    Displays its own errors via st.error() for validation/pipeline
    failures, so the caller doesn't need to handle VideoProcessingError.
    """
    st.markdown("### ارفعي فيديو للإشارة")
    st.caption(
        f"الصيغ المدعومة: MP4, AVI, MOV, MKV — الحد الأقصى للحجم {MAX_UPLOAD_SIZE_MB} ميجابايت"
    )

    uploaded_file = st.file_uploader(
        label="اختاري فيديو",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        return None

    st.video(uploaded_file)

    translate_clicked = st.button("🤟 ترجمي الإشارة", type="primary", use_container_width=True)

    if not translate_clicked:
        return None

    file_bytes = uploaded_file.getvalue()
    original_filename = uploaded_file.name

    with st.spinner("بنحلل حركة الإشارة... ممكن ياخد شوية ثواني"):
        try:
            result = video_service.process_uploaded_video(file_bytes, original_filename)
        except VideoProcessingError as error:
            st.error(str(error))
            return None
        except Exception as error:
            # Unexpected/unhandled failure -- log-worthy, but still show
            # something sane to the user instead of a raw traceback.
            st.error("حصل خطأ غير متوقع أثناء معالجة الفيديو. جربي تاني.")
            st.exception(error)  # visible only if Streamlit debug/dev mode
            return None

    return result
