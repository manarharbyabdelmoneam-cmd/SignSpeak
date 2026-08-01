"""
Services: Model Service
========================
Owns the full ML pipeline lifecycle: LandmarkExtractor (MediaPipe) +
SequenceBuilder (normalization/resampling) + Predictor (Keras model).

Provides one simple entry point - process_video() - so the UI layer
never touches MediaPipe or Keras directly. This module has no Streamlit
imports, so it can be wrapped with st.cache_resource in the UI, or reused
in tests / a future API unchanged.
"""

import logging
from pathlib import Path
from typing import Union

from src.core.landmark_extractor import LandmarkExtractor
from src.core.sequence_builder import SequenceBuilder
from src.core.predictor import Predictor

from config.settings import MODELS_DIR, SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

# MediaPipe's own hand/pose detector files (not the trained SignBridge
# model). These must be downloaded separately - see the note below.
HAND_MODEL_FILENAME = "hand_landmarker.task"
POSE_MODEL_FILENAME = "pose_landmarker_full.task"


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
        hand_model_path = models_dir / HAND_MODEL_FILENAME
        pose_model_path = models_dir / POSE_MODEL_FILENAME

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
        if it's unreadable / has no usable gesture (too few frames, no
        active segment). Callers (video_service.py) should catch
        ValueError and show a friendly "couldn't detect a sign, try
        another video" message instead of crashing.
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
