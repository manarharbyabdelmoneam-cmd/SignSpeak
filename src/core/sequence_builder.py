"""
Core: Sequence Builder
=======================
Takes the raw, un-normalized landmark sequence produced by
LandmarkExtractor.extract_video_sequence() -> shape (N, 153) -- and turns
it into the exact (80, 153) normalized, temporally-resampled sequence
the model expects.

Pipeline (must match training exactly):
    1. Detect the active gesture segment (frames where a hand is visible)
    2. Normalize landmarks relative to the shoulder center/scale
    3. Temporally resample to a fixed length (80 time steps)

This mirrors `final_preprocess_video()` from the training and external-
validation preprocessing notebooks, minus the landmark extraction step
itself (that lives in LandmarkExtractor).
"""

import numpy as np


class SequenceBuilder:
    """Builds a fixed-length, normalized landmark sequence for the model."""

    LEFT_HAND_SIZE = 63
    RIGHT_HAND_SIZE = 63
    UPPER_BODY_SIZE = 27
    TOTAL_FEATURES = LEFT_HAND_SIZE + RIGHT_HAND_SIZE + UPPER_BODY_SIZE  # 153

    # Indices within the 9-point upper-body block (after reshape to (9, 3))
    LEFT_SHOULDER_IDX = 1
    RIGHT_SHOULDER_IDX = 2

    def __init__(self, target_length: int = 80):
        self.target_length = target_length

    # ------------------------------------------------------------
    # Step 1: Active gesture segmentation
    # ------------------------------------------------------------
    @staticmethod
    def extract_active_gesture_segment(video_features: np.ndarray):
        """
        Trim the sequence to the span between the first and last frame
        where at least one hand was detected.

        Returns (active_segment, gesture_start, gesture_end).
        If no hand is detected in any frame, returns the full sequence
        unchanged (matches training notebook behavior).
        """
        detected_frames = []

        for idx, frame_features in enumerate(video_features):
            left_hand = frame_features[:63]
            right_hand = frame_features[63:126]

            left_detected = not np.all(left_hand == 0)
            right_detected = not np.all(right_hand == 0)

            if left_detected or right_detected:
                detected_frames.append(idx)

        if len(detected_frames) == 0:
            return video_features, 0, len(video_features) - 1

        gesture_start = detected_frames[0]
        gesture_end = detected_frames[-1]

        active_segment = video_features[gesture_start:gesture_end + 1]

        return active_segment, gesture_start, gesture_end

    # ------------------------------------------------------------
    # Step 2: Normalization
    # ------------------------------------------------------------
    @classmethod
    def normalize_frame_features(cls, frame_features: np.ndarray) -> np.ndarray:
        """
        Normalize a single frame's 153 features relative to the signer's
        upper body:
            - center = midpoint between shoulders
            - scale  = XY distance between shoulders

        Missing hands (all-zero) are left untouched (not shifted/scaled),
        matching training behavior exactly.
        """
        frame_features = frame_features.copy()

        left_hand = frame_features[:63].reshape(21, 3)
        right_hand = frame_features[63:126].reshape(21, 3)
        upper_body = frame_features[126:153].reshape(9, 3)

        # upper_body order: 0=Nose, 1=L.Shoulder, 2=R.Shoulder, 3=L.Elbow,
        # 4=R.Elbow, 5=L.Wrist, 6=R.Wrist, 7=L.Hip, 8=R.Hip
        left_shoulder = upper_body[cls.LEFT_SHOULDER_IDX]
        right_shoulder = upper_body[cls.RIGHT_SHOULDER_IDX]

        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        shoulder_distance = np.linalg.norm(
            left_shoulder[:2] - right_shoulder[:2]
        )
        shoulder_distance = max(shoulder_distance, 1e-6)  # avoid division by zero

        upper_body_normalized = (upper_body - shoulder_center) / shoulder_distance

        if not np.all(left_hand == 0):
            left_hand_normalized = (left_hand - shoulder_center) / shoulder_distance
        else:
            left_hand_normalized = left_hand

        if not np.all(right_hand == 0):
            right_hand_normalized = (right_hand - shoulder_center) / shoulder_distance
        else:
            right_hand_normalized = right_hand

        return np.concatenate([
            left_hand_normalized.flatten(),
            right_hand_normalized.flatten(),
            upper_body_normalized.flatten(),
        ])

    @classmethod
    def normalize_sequence(cls, sequence: np.ndarray) -> np.ndarray:
        """Apply normalize_frame_features to every frame in the sequence."""
        normalized = [cls.normalize_frame_features(frame) for frame in sequence]
        return np.asarray(normalized, dtype=np.float32)

    # ------------------------------------------------------------
    # Step 3: Temporal resampling
    # ------------------------------------------------------------
    @staticmethod
    def temporal_resample(sequence: np.ndarray, target_length: int = 80) -> np.ndarray:
        """
        Resample a (T, 153) sequence to (target_length, 153) using linear
        interpolation along the time axis. Matches training exactly
        (equivalent to the notebook's np.interp-based implementation).
        """
        sequence = np.asarray(sequence, dtype=np.float32)
        original_length, num_features = sequence.shape

        if original_length == target_length:
            return sequence.copy()

        if original_length < 2:
            raise ValueError("Temporal resampling requires at least 2 time steps.")

        original_time = np.linspace(0.0, 1.0, original_length)
        target_time = np.linspace(0.0, 1.0, target_length)

        resampled = np.empty((target_length, num_features), dtype=np.float32)

        for feature_idx in range(num_features):
            resampled[:, feature_idx] = np.interp(
                target_time, original_time, sequence[:, feature_idx]
            )

        return resampled

    # ------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------
    def build(self, video_features: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Run the full sequence-building pipeline on raw per-frame features
        from LandmarkExtractor.extract_video_sequence().

        Returns (final_sequence, metadata) where final_sequence has shape
        (target_length, 153).

        Raises ValueError if the video doesn't contain enough valid frames
        or the output fails validation -- mirrors final_preprocess_video()
        in the training notebook so failures behave identically.
        """
        if len(video_features) < 2:
            raise ValueError("Video contains insufficient valid frames.")

        active_segment, gesture_start, gesture_end = self.extract_active_gesture_segment(
            video_features
        )

        if len(active_segment) < 2:
            raise ValueError("Active gesture segment is too short.")

        normalized_segment = self.normalize_sequence(active_segment)

        final_sequence = self.temporal_resample(
            normalized_segment, target_length=self.target_length
        )

        if final_sequence.shape != (self.target_length, self.TOTAL_FEATURES):
            raise ValueError(f"Unexpected final shape: {final_sequence.shape}")

        if np.isnan(final_sequence).any():
            raise ValueError("NaN values detected in the final sequence.")

        if np.isinf(final_sequence).any():
            raise ValueError("Infinite values detected in the final sequence.")

        metadata = {
            "original_frames": len(video_features),
            "gesture_start": gesture_start,
            "gesture_end": gesture_end,
            "active_frames": len(active_segment),
        }

        return final_sequence, metadata
