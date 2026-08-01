"""
Core: Predictor
================
Loads the trained SignBridge AI model (Keras BiLSTM) and turns a single
(80, 153) normalized landmark sequence into a predicted Arabic/English
phrase with a confidence score.

Mirrors the inference contract recorded in the training notebook's
`deployment_config.json`:
    - input_shape: (80, 153), dtype float32
    - confidence_threshold: 0.60
    - low_confidence_action: ask the user to repeat the sign
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
from tensorflow import keras

from config.settings import (
    MODEL_PATH,
    TFLITE_MODEL_PATH,
    USE_TFLITE,
    LABEL_MAPPING_PATH,
    CONFIDENCE_THRESHOLD,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
)


class Predictor:
    """Runs inference on a single preprocessed landmark sequence."""

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        label_mapping_path: Union[str, Path] = LABEL_MAPPING_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        use_tflite: bool = USE_TFLITE,
    ):
        self.confidence_threshold = confidence_threshold
        self.use_tflite = use_tflite
        self.expected_shape = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)

        self._labels = self._load_labels(label_mapping_path)

        if self.use_tflite:
            self._interpreter = self._load_tflite_model(TFLITE_MODEL_PATH)
        else:
            self._model = self._load_keras_model(model_path)

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------
    @staticmethod
    def _load_keras_model(model_path: Union[str, Path]):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Make sure 'signbridge_best_model.keras' is present in "
                "the models/ directory (or set MODEL_DOWNLOAD_URL)."
            )
        return keras.models.load_model(model_path)

    @staticmethod
    def _load_tflite_model(tflite_path: Union[str, Path]):
        import tensorflow as tf  # local import: only needed for TFLite path

        tflite_path = Path(tflite_path)
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite model file not found: {tflite_path}")

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        return interpreter

    @staticmethod
    def _load_labels(label_mapping_path: Union[str, Path]) -> dict:
        label_mapping_path = Path(label_mapping_path)
        if not label_mapping_path.exists():
            raise FileNotFoundError(f"Label mapping file not found: {label_mapping_path}")

        with open(label_mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data["labels"]  # {"0": {"en": ..., "ar": ...}, ...}

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    def _validate_input(self, sequence: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.shape != self.expected_shape:
            raise ValueError(
                f"Expected input shape {self.expected_shape}, "
                f"received {sequence.shape}"
            )

        if np.isnan(sequence).any() or np.isinf(sequence).any():
            raise ValueError("Input sequence contains NaN or infinite values.")

        return sequence

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    def _predict_probabilities(self, sequence: np.ndarray) -> np.ndarray:
        """Returns the raw softmax probability vector for one sequence."""
        batch = np.expand_dims(sequence, axis=0)  # (1, 80, 153)

        if self.use_tflite:
            input_details = self._interpreter.get_input_details()
            output_details = self._interpreter.get_output_details()

            self._interpreter.set_tensor(input_details[0]["index"], batch)
            self._interpreter.invoke()
            probabilities = self._interpreter.get_tensor(output_details[0]["index"])[0]
        else:
            probabilities = self._model.predict(batch, verbose=0)[0]

        return probabilities

    def predict(self, sequence: np.ndarray) -> dict:
        """
        Predict the sign phrase for a single (80, 153) landmark sequence.

        Returns a dict:
            {
                "predicted_index": int,
                "predicted_label_en": str,
                "predicted_label_ar": str,
                "confidence": float,
                "is_confident": bool,           # confidence >= threshold
                "top_3": [ {index, en, ar, confidence}, ... ],
            }

        When `is_confident` is False, the UI should ask the user to
        repeat the sign rather than trusting the prediction.
        """
        sequence = self._validate_input(sequence)
        probabilities = self._predict_probabilities(sequence)

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3 = [
            {
                "index": int(idx),
                "en": self._labels[str(idx)]["en"],
                "ar": self._labels[str(idx)]["ar"],
                "confidence": float(probabilities[idx]),
            }
            for idx in top_3_indices
        ]

        predicted_label = self._labels[str(predicted_index)]

        return {
            "predicted_index": predicted_index,
            "predicted_label_en": predicted_label["en"],
            "predicted_label_ar": predicted_label["ar"],
            "confidence": confidence,
            "is_confident": confidence >= self.confidence_threshold,
            "top_3": top_3,
        }
