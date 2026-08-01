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
MODEL_FILENAME = "arsl_gru_bilstm.h5"          # عدّليه لو اسم الملف مختلف
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# لو هنحمل الموديل من الإنترنت وقت الـ runtime بدل ما يتخزن في الـ repo
MODEL_DOWNLOAD_URL = os.environ.get("MODEL_DOWNLOAD_URL", "")

# ============================================================
# Labels
# ============================================================
LABEL_MAPPING_PATH = CONFIG_DIR / "label_mapping.json"

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

# عدد الإحداثيات المستخرجة لكل frame (حسب اللاندماركس اللي بتستخدميها)
# Pose: 33 point * 4 (x,y,z,visibility) = 132
# Face: 468 point * 3 (x,y,z) = 1404  (لو مستخدمة)
# Left hand: 21 point * 3 = 63
# Right hand: 21 point * 3 = 63
USE_FACE_LANDMARKS = False   # غيّريها لو محتاجة الوش في الفيتشرز
LANDMARKS_PER_FRAME = 132 + 63 + 63 + (1404 if USE_FACE_LANDMARKS else 0)

# ============================================================
# Sequence / Video Processing
# ============================================================
SEQUENCE_LENGTH = 30            # عدد الفريمات في كل sequence تتبعت للموديل
TARGET_FPS = 30                 # الـ FPS المعياري (لتوحيد الفيديوهات المختلفة)
FRAME_RESIZE_DIM = (224, 224)   # لو محتاجة تعملي resize للفريمات

# حد أقصى لمدة الفيديو المرفوع (بالثواني) عشان تتحكمي في الـ processing time
MAX_VIDEO_DURATION_SEC = 15

# ============================================================
# Prediction
# ============================================================
CONFIDENCE_THRESHOLD = 0.6      # تحت الرقم ده الموديل يقول "مش متأكد"

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
