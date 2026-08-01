"""
Global configuration for SignBridge AI.
All paths, constants, and thresholds used across the project live here.
"""

import os
from pathlib import Path

# ============================================================
# Base Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # signbridge-ai/

MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime-created dirs exist (models/logs may not be in the repo)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Model Files
# ============================================================
MODEL_FILENAME = "signbridge_best_model.keras"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# نسخة TFLite اختيارية (أخف وزنا، للـ deployment المحدود بالـ resources)
TFLITE_MODEL_FILENAME = "signbridge_best_model.tflite"
TFLITE_MODEL_PATH = MODELS_DIR / TFLITE_MODEL_FILENAME
USE_TFLITE = False   # غيّريها True لو هتستخدمي نسخة الـ TFLite بدل الـ Keras

MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
DEPLOYMENT_CONFIG_PATH = MODELS_DIR / "deployment_config.json"

# لو هنحمل الموديل من الإنترنت وقت الـ runtime بدل ما يتخزن في الـ repo
# (الملف حجمه ~672 KB بس، فمش لازم غالبا، بس سايبها موجودة للمرونة)
MODEL_DOWNLOAD_URL = os.environ.get("MODEL_DOWNLOAD_URL", "")

# ============================================================
# Labels
# ============================================================
LABEL_MAPPING_PATH = CONFIG_DIR / "label_mapping.json"
NUM_CLASSES = 12

# ============================================================
# MediaPipe Holistic Settings
# ============================================================
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# بيستخدم كل الـ landmarks (Pose + Face + Left Hand + Right Hand)
USE_FACE_LANDMARKS = True
USE_POSE_LANDMARKS = True
USE_HAND_LANDMARKS = True

# ============================================================
# Sequence / Video Processing
# ============================================================
# لازم يتطابقوا بالظبط مع شكل بيانات التدريب: (80, 153)
SEQUENCE_LENGTH = 80            # عدد الفريمات في كل sequence تتبعت للموديل
FEATURES_PER_FRAME = 153        # بعد الـ normalization والـ feature engineering
INPUT_SHAPE = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)

TARGET_FPS = 30                 # الـ FPS المعياري (لتوحيد الفيديوهات المختلفة قبل استخراج 80 frame)
PADDING_VALUE = 0.0             # الموديل بيستخدم Masking بـ mask_value=0.0

# حد أقصى لمدة الفيديو المرفوع (بالثواني) عشان تتحكمي في الـ processing time
MAX_VIDEO_DURATION_SEC = 15

# ============================================================
# Prediction
# ============================================================
CONFIDENCE_THRESHOLD = 0.60     # تحت الرقم ده الموديل يطلب من المستخدم يعيد الإشارة
LOW_CONFIDENCE_ACTION = "Request the user to repeat the sign"

# ============================================================
# Streamlit App
# ============================================================
APP_TITLE = "SignBridge AI"
APP_ICON = "🤟"
PAGE_LAYOUT = "wide"

MAX_UPLOAD_SIZE_MB = 200         # الحد الافتراضي بتاع Streamlit Cloud

# ============================================================
# Logging
# ============================================================
LOG_FILE_PATH = LOGS_DIR / "signbridge.log"
LOG_LEVEL = "INFO"
