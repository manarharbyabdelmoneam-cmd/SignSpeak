"""
Services: Model Service
========================
Owns the full ML pipeline lifecycle: LandmarkExtractor (MediaPipe) +
SequenceBuilder (normalization/resampling) + Predictor (Keras model).

Provides one simple entry point - process_video() - so the UI layer
never touches MediaPipe or Keras directly. This module has no Streamlit
imports, so it can be wrapped with st.cache_resource in the UI, or reused
in tests / a future API unchanged.

MediaPipe's own Hand/Pose detector files (*.task) are downloaded
automatically on first run if missing, instead of being committed to
the repo -- keeps the repo lean and avoids Git LFS.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Union

from src.core.landmark_extractor import LandmarkExtractor
from src.core.sequence_builder import SequenceBuilder
from src.core.predictor import Predictor

from config.settings import MODELS_DIR, SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

HAND_MODEL_FILENAME = "hand_landmarker.task"
POSE_MODEL_FILENAME = "pose_landmarker_full.task"

# Official MediaPipe model URLs (Google Cloud Storage, stable versioned paths)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def _ensure_model_file(path: Path, url: str) -> Path:
    """Download a MediaPipe model file to `path` if it doesn't exist yet."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe model: %s -> %s", url, path)

    try:
        urllib.request.urlretrieve(url, path)
    except Exception as error:
        raise RuntimeError(
            f"Failed to download required MediaPipe model from {url}. "
            f"Check network access to storage.googleapis.com. Original error: {error}"
        ) from error

    logger.info("Downloaded MediaPipe model successfully: %s", path.name)
    return path


class ModelService:
    """
    Loads all ML components once and exposes process_video().

    Usage:
        service = ModelService()
        result = service.process_video("path/to/video.mp4")
        service.close()   # or use as a context manager
    """

    def __init__(
        self,
        models_dir: Union[str, Path] = MODELS_DIR,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        models_dir = Path(models_dir)

        hand_model_path = _ensure_model_file(
            models_dir / HAND_MODEL_FILENAME, HAND_MODEL_URL
        )
        pose_model_path = _ensure_model_file(
            models_dir / POSE_MODEL_FILENAME, POSE_MODEL_URL
        )

        logger.info("Loading LandmarkExtractor (MediaPipe Hand + Pose)...")
        self.landmark_extractor = LandmarkExtractor(
            hand_model_path=hand_model_path,
            pose_model_path=pose_model_path,
        )

        logger.info("Initializing SequenceBuilder...")
        self.sequence_builder = SequenceBuilder(target_length=sequence_length)

        logger.info("Loading Predictor (Keras model + labels)...")
        self.predictor = Predictor()

        logger.info("ModelService ready.")

    # ------------------------------------------------------------
    def process_video(self, video_path: Union[str, Path]) -> dict:
        """
        Full pipeline: video file -> landmarks -> normalized sequence -> prediction.

        Returns the same dict shape as Predictor.predict(), plus:
            "metadata": {
                "original_frames": int,
                "gesture_start": int,
                "gesture_end": int,
                "active_frames": int,
            }

        Raises FileNotFoundError if the video doesn't exist, or ValueError
        if it's unreadable / has no usable gesture. Callers
        (video_service.py) should catch ValueError and show a friendly
        "couldn't detect a sign, try another video" message.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        raw_features = self.landmark_extractor.extract_video_sequence(video_path)
        final_sequence, metadata = self.sequence_builder.build(raw_features)

        result = self.predictor.predict(final_sequence)
        result["metadata"] = metadata

        return result

    def close(self):
        """Release MediaPipe resources (call on app shutdown)."""
        self.landmark_extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
