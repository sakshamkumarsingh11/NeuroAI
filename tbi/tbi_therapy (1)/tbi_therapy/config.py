"""Central configuration. Edit paths here, not in module code."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Paths ----
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SESSIONS_DIR = DATA_DIR / "sessions"
INSTANCE_DIR = BASE_DIR / "instance"

CMU_DICT_PATH = DATA_DIR / "cmudict-0.7b"
PPO_ACTOR_PATH = MODELS_DIR / "ppo_actor.pt"
PPO_CRITIC_PATH = MODELS_DIR / "ppo_critic.pt"

DATABASE_PATH = INSTANCE_DIR / "therapy.db"

# ---- Audio ----
SAMPLE_RATE = 16000
AUDIO_UPLOAD_DIR = DATA_DIR / "uploads"

# ---- Severity thresholds (from Phase 2) ----
SEVERITY_SCORES = {
    "Severe": 5,
    "Moderate": 3,
    "Mild": 1,
    "Normal": 0,
}

# ---- RL Planner ----
RL_CONFIG = {
    "gamma": 0.99,
    "epsilon": 0.2,
    "learning_rate": 0.0003,
    "hidden_size": 128,
    "state_dim": 5,      # severity, recent_acc, trend, session_count, fatigue
    "action_dim": 3,     # easier, same, harder
}

# ---- MediaPipe ----
FACE_MESH_CONFIG = {
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# Facial landmarks of interest for TBI (MediaPipe indices)
LIP_LANDMARKS = {
    "upper_outer": 13,
    "lower_outer": 14,
    "upper_inner": 12,
    "lower_inner": 15,
    "left_corner": 61,
    "right_corner": 291,
}

JAW_LANDMARK = 152  # chin tip
FOREHEAD_LANDMARK = 10

# Symmetry reference points (left, right pairs)
SYMMETRY_PAIRS = [
    (61, 291),   # mouth corners
    (33, 263),   # outer eye corners
    (234, 454),  # cheek edges
]

# ---- Flask ----
FLASK_CONFIG = {
    "DEBUG": True,
    "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-key-change-in-production"),
    "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,  # 16 MB audio uploads
}

# Ensure dirs exist at import time
for d in [DATA_DIR, MODELS_DIR, SESSIONS_DIR, INSTANCE_DIR, AUDIO_UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)
