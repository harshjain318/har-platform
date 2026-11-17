"""Central configuration constants for the HAR platform."""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "uploads"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
DATASETS_DIR = ROOT / "datasets"

# Unified signal format used by every model in the platform
TARGET_FS = 50.0          # Hz
WINDOW = 128              # samples (2.56 s at 50 Hz)
OVERLAP = 0.5
STRIDE = int(WINDOW * (1 - OVERLAP))   # 64 samples
LOW_CONFIDENCE = 0.60     # windows with max-probability below this are flagged
GRAVITY = 9.80665         # m/s^2

# Internal units: accelerometer in g, gyroscope in rad/s
ACC_UNITS = {"g": 1.0, "m/s2": 1.0 / GRAVITY, "mg": 1e-3}
GYRO_UNITS = {"rad/s": 1.0, "deg/s": math.pi / 180.0}

CANONICAL_ACTIVITIES = [
    "walking", "upstairs", "downstairs", "sitting", "standing", "running", "lying",
]
ACTIVITY_DISPLAY = {
    "walking": "Walking", "upstairs": "Walking Upstairs", "downstairs": "Walking Downstairs",
    "sitting": "Sitting", "standing": "Standing", "running": "Running", "lying": "Lying",
}
DATASET_SOURCES = ["UCI HAR", "WISDM", "MotionSense", "HHAR", "Custom"]
CHANNELS_ACC = ["acc_x", "acc_y", "acc_z"]
CHANNELS_GYRO = ["gyro_x", "gyro_y", "gyro_z"]


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, RESULTS_DIR, CHECKPOINT_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
