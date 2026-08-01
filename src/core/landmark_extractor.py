"""
Core: Landmark Extractor
=========================
Extracts a fixed 153-value landmark feature vector per frame, using the
exact same MediaPipe Tasks pipeline used during model training:

    left_hand(63) + right_hand(63) + upper_body_pose(27) = 153

IMPORTANT: This must stay byte-for-byte consistent with the training
notebook's `extract_frame_features`. Any change here (landmark indices,
feature order, missing-value handling) will silently break inference,
since the model was trained on this exact feature layout.
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import mediapipe as mp


class LandmarkExtractor:
    """Extracts hand + upper-body pose landmarks from video frames."""

    # Upper-body pose landmark indices from MediaPipe Pose (BlazePose 33-point)
    # 0=Nose, 11/12=Shoulders, 13/14=Elbows, 15/16=Wrists, 23/24=Hips
    UPPER_BODY_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]

    LEFT_HAND_SIZE = 21 * 3   # 63
    RIGHT_HAND_SIZE = 21 * 3  # 63
    UPPER_BODY_SIZE = 9 * 3   # 27
    TOTAL_FEATURES = LEFT_HAND_SIZE + RIGHT_HAND_SIZE + UPPER_BODY_SIZE  # 153

    FRAME_SIZE = (224, 224)  # (width, height) used during training

    def __init__(
        self,
        hand_model_path: Union[str, Path],
        pose_model_path: Union[str, Path],
        num_hands: int = 2,
    ):
        hand_model_path = str(hand_model_path)
        pose_model_path = str(pose_model_path)

        if not Path(hand_model_path).exists():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {hand_model_path}\n"
                "Download 'hand_landmarker.task' from the MediaPipe model "
                "zoo and place it in the models/ directory."
            )
        if not Path(pose_model_path).exists():
            raise FileNotFoundError(
                f"Pose Landmarker model not found: {pose_model_path}\n"
                "Download 'pose_landmarker_full.task' from the MediaPipe "
                "model zoo and place it in the models/ directory."
            )

        base_options_cls = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options_cls(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options_cls(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    # ------------------------------------------------------------
    # Frame-level preprocessing (must match training exactly)
    # ------------------------------------------------------------
    @staticmethod
    def resize_with_padding(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        """Aspect-ratio-preserving resize with black padding, matching training."""
        target_w, target_h = target_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    # ------------------------------------------------------------
    # Single-frame feature extraction
    # ------------------------------------------------------------
    def extract_frame_features(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract a (153,) float32 feature vector from a single RGB frame.

        `frame_rgb` must already be RGB (not BGR) and resized via
        `resize_with_padding` to FRAME_SIZE, exactly as in training.

        Layout:
            [0:63]    left hand  (21 landmarks x xyz)
            [63:126]  right hand (21 landmarks x xyz)
            [126:153] upper-body pose (9 landmarks x xyz)

        Missing hand or pose detections are zero-filled (not interpolated
        here - that happens later, if at all, in sequence_builder.py).
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_result = self._hand_landmarker.detect(mp_image)
        pose_result = self._pose_landmarker.detect(mp_image)

        left_hand = np.zeros(self.LEFT_HAND_SIZE, dtype=np.float32)
        right_hand = np.zeros(self.RIGHT_HAND_SIZE, dtype=np.float32)
        upper_body = np.zeros(self.UPPER_BODY_SIZE, dtype=np.float32)

        for hand_landmarks, handedness in zip(
            hand_result.hand_landmarks, hand_result.handedness
        ):
            vector = []
            for landmark in hand_landmarks:
                vector.extend([landmark.x, landmark.y, landmark.z])
            vector = np.array(vector, dtype=np.float32)

            hand_label = handedness[0].category_name
            if hand_label == "Left":
                left_hand = vector
            elif hand_label == "Right":
                right_hand = vector

        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            body_vector = []
            for idx in self.UPPER_BODY_INDICES:
                landmark = pose_landmarks[idx]
                body_vector.extend([landmark.x, landmark.y, landmark.z])
            upper_body = np.array(body_vector, dtype=np.float32)

        return np.concatenate([left_hand, right_hand, upper_body])

    # ------------------------------------------------------------
    # Full-video extraction (raw, un-normalized, un-resampled)
    # ------------------------------------------------------------
    def extract_video_sequence(self, video_path: Union[str, Path]) -> np.ndarray:
        """
        Read every original frame of the video and extract its 153-feature
        vector. Returns shape (num_original_frames, 153).

        This is intentionally "raw": no active-gesture segmentation,
        normalization, or temporal resampling. Those live in
        sequence_builder.py so this class stays a pure landmark extractor.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        features = []
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.resize_with_padding(frame, self.FRAME_SIZE)

                features.append(self.extract_frame_features(frame))
        finally:
            cap.release()

        if not features:
            raise ValueError(f"No frames could be read from video: {video_path}")

        return np.asarray(features, dtype=np.float32)

    def close(self):
        """Release MediaPipe resources. Call when done, or use as a context manager."""
        self._hand_landmarker.close()
        self._pose_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
