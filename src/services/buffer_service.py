"""
Services: Buffer Service
==========================
Manages a rolling buffer of landmark features for FUTURE live-camera
translation (webcam -> continuous frames -> periodic prediction).

NOT wired into the MVP UI. We agreed the MVP ships upload-only (no
streamlit-webrtc / live camera) because of Streamlit Cloud free-tier
limitations. This file exists so the architecture stays consistent and
so live translation can be added later without redesigning the services
layer - it isn't imported by any page yet.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from src.core.sequence_builder import SequenceBuilder
from src.core.predictor import Predictor
from config.settings import SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class BufferService:
    """
    Accumulates per-frame 153-feature landmark vectors from a live stream
    and produces a prediction once enough frames are collected.

    Intended usage (future live-camera page):
        buffer = BufferService(predictor)
        for frame in webcam_stream:
            features = landmark_extractor.extract_frame_features(frame)
            result = buffer.add_frame(features)
            if result is not None:
                show_prediction(result)
    """

    def __init__(
        self,
        predictor: Predictor,
        window_size: int = SEQUENCE_LENGTH,
        stride: int = 10,
        min_active_frames: int = 2,
    ):
        """
        window_size: number of raw frames to accumulate before building
                     a sequence and running prediction (a sliding window,
                     not necessarily == model's fixed SEQUENCE_LENGTH,
                     since SequenceBuilder resamples anyway).
        stride: how many new frames must arrive before we re-predict
                (avoids running inference on every single frame).
        min_active_frames: minimum frames with a detected hand required
                            before we bother running the pipeline.
        """
        self.predictor = predictor
        self.sequence_builder = SequenceBuilder(target_length=SEQUENCE_LENGTH)
        self.window_size = window_size
        self.stride = stride
        self.min_active_frames = min_active_frames

        self._frames: deque = deque(maxlen=window_size)
        self._frames_since_last_prediction = 0

    # ------------------------------------------------------------
    def add_frame(self, frame_features: np.ndarray) -> Optional[dict]:
        """
        Add one 153-feature frame vector to the buffer.

        Returns a prediction dict (same shape as Predictor.predict())
        once `stride` new frames have accumulated since the last
        prediction AND the buffer holds enough active (hand-visible)
        frames. Otherwise returns None (not enough data yet, or a
        no-op tick between predictions).
        """
        self._frames.append(frame_features)
        self._frames_since_last_prediction += 1

        if len(self._frames) < 2:
            return None

        if self._frames_since_last_prediction < self.stride:
            return None

        buffered_sequence = np.asarray(self._frames, dtype=np.float32)

        active_count = self._count_active_frames(buffered_sequence)
        if active_count < self.min_active_frames:
            return None

        try:
            final_sequence, metadata = self.sequence_builder.build(buffered_sequence)
        except ValueError as error:
            logger.debug("Buffer not ready for prediction yet: %s", error)
            return None

        self._frames_since_last_prediction = 0

        result = self.predictor.predict(final_sequence)
        result["metadata"] = metadata
        return result

    @staticmethod
    def _count_active_frames(sequence: np.ndarray) -> int:
        left_hand = sequence[:, :63]
        right_hand = sequence[:, 63:126]
        left_active = ~np.all(left_hand == 0, axis=1)
        right_active = ~np.all(right_hand == 0, axis=1)
        return int(np.sum(left_active | right_active))

    def reset(self) -> None:
        """Clear the buffer, e.g. after a confident prediction is shown."""
        self._frames.clear()
        self._frames_since_last_prediction = 0

    def is_ready(self) -> bool:
        """True once the buffer holds enough frames to attempt a prediction."""
        return len(self._frames) >= 2
