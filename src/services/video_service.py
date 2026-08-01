"""
Services: Video Service
=========================
Handles everything around a user-uploaded video file: validation, saving
to a temp path, running it through ModelService, cleanup, and turning any
pipeline failure into a friendly Arabic message for the UI.

ui/pages/upload_page.py should only ever talk to VideoService - never to
ModelService, LandmarkExtractor, etc. directly.
"""

import logging
import tempfile
from pathlib import Path

import cv2

from src.services.model_service import ModelService
from config.settings import MAX_VIDEO_DURATION_SEC, MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class VideoProcessingError(Exception):
    """
    Raised for any user-facing video problem. The message is written in
    Egyptian Arabic and is safe to display directly in the UI (e.g. via
    st.error(str(error))).
    """
    pass


class VideoService:
    """Orchestrates validation + preprocessing + prediction for one upload."""

    def __init__(self, model_service: ModelService):
        self.model_service = model_service

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    def _validate_file_basics(self, filename: str, file_size_bytes: int) -> None:
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise VideoProcessingError(
                f"صيغة الفيديو '{extension}' مش مدعومة. "
                f"الصيغ المدعومة: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size_bytes > max_bytes:
            raise VideoProcessingError(
                f"حجم الفيديو أكبر من الحد المسموح ({MAX_UPLOAD_SIZE_MB} ميجابايت). "
                "جربي تضغطي الفيديو أو ترفعي مقطع أقصر."
            )

    def _validate_duration(self, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration_sec = (frame_count / fps) if fps > 0 else 0
        finally:
            cap.release()

        if duration_sec == 0:
            raise VideoProcessingError(
                "مش قادرين نفتح الفيديو ده. اتأكدي إن الملف سليم وجربي تاني."
            )

        if duration_sec > MAX_VIDEO_DURATION_SEC:
            raise VideoProcessingError(
                f"الفيديو طويل أكتر من اللازم ({duration_sec:.1f} ثانية). "
                f"الحد الأقصى المسموح {MAX_VIDEO_DURATION_SEC} ثانية."
            )

    # ------------------------------------------------------------
    # Temp file handling
    # ------------------------------------------------------------
    def _save_to_temp_file(self, file_bytes: bytes, original_filename: str) -> Path:
        """
        Save uploaded bytes to a temp file, preserving the original
        extension (OpenCV needs it to pick the right codec/container).
        """
        extension = Path(original_filename).suffix.lower()
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=extension, prefix="signbridge_upload_"
        )
        try:
            temp_file.write(file_bytes)
        finally:
            temp_file.close()

        return Path(temp_file.name)

    # ------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------
    def process_uploaded_video(self, file_bytes: bytes, original_filename: str) -> dict:
        """
        Full flow for one uploaded video:
            validate basics -> save temp file -> validate duration
            -> run ML pipeline -> cleanup temp file -> return result

        Returns the same dict shape as ModelService.process_video()
        (predicted_label_en/ar, confidence, is_confident, top_3, metadata).

        Raises VideoProcessingError with an Arabic message safe to show
        directly in the UI. Any other exception is a real bug and should
        propagate (don't swallow it silently).
        """
        self._validate_file_basics(original_filename, len(file_bytes))

        temp_path = self._save_to_temp_file(file_bytes, original_filename)
        logger.info("Saved uploaded video to temp path: %s", temp_path)

        try:
            self._validate_duration(temp_path)

            try:
                result = self.model_service.process_video(temp_path)
            except ValueError as error:
                logger.warning("Pipeline rejected video %s: %s", original_filename, error)
                raise VideoProcessingError(
                    "مش قادرين نتعرف على إشارة واضحة في الفيديو ده. "
                    "جربي تصوري تاني بإضاءة كويسة، وإن الإشارة والإيدين واضحين قدام الكاميرا."
                ) from error

            logger.info(
                "Prediction for %s: %s (confidence=%.2f)",
                original_filename,
                result["predicted_label_en"],
                result["confidence"],
            )
            return result

        finally:
            temp_path.unlink(missing_ok=True)
