"""MediaPipe-based facial analysis for TBI.

Extracts three clinically useful signals from a single image frame:
  1. Lip closure score (poor closure = motor weakness)
  2. Jaw vertical position (reduced movement = weakness)
  3. Facial asymmetry (droop = common in TBI/stroke)
"""
import base64
import io
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import MODELS_DIR, LIP_LANDMARKS, JAW_LANDMARK, FOREHEAD_LANDMARK, SYMMETRY_PAIRS

# New Tasks API (mediapipe >= 0.10.30)
_BaseOptions = mp.tasks.BaseOptions
_FaceLandmarker = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
_VisionRunningMode = mp.tasks.vision.RunningMode

_MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")

_face_landmarker = None


def get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        options = _FaceLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=_VisionRunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _face_landmarker = _FaceLandmarker.create_from_options(options)
    return _face_landmarker


# ---------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------
def _landmarks_to_np(face_landmarks, width, height):
    """Convert MediaPipe NormalizedLandmark list to (N, 3) numpy array in pixel coords."""
    return np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in face_landmarks],
        dtype=np.float32,
    )


def analyze_frame(image_bgr: np.ndarray) -> Optional[dict]:
    """
    Run MediaPipe on one BGR image, compute TBI-relevant metrics.

    Returns None if no face detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = get_face_landmarker().detect(mp_image)

    if not result.face_landmarks:
        return None

    pts = _landmarks_to_np(result.face_landmarks[0], w, h)
    face_height = np.linalg.norm(pts[FOREHEAD_LANDMARK] - pts[JAW_LANDMARK])
    if face_height < 1e-3:
        return None

    # ---- Lip closure ----
    upper_lip = pts[LIP_LANDMARKS["upper_inner"]]
    lower_lip = pts[LIP_LANDMARKS["lower_inner"]]
    lip_gap = np.linalg.norm(upper_lip - lower_lip)
    # Normalize by face height (makes it scale-invariant)
    lip_gap_norm = lip_gap / face_height
    # Closure score: 1.0 = fully closed, 0.0 = wide open
    # Typical closed mouth has lip_gap_norm < 0.015
    lip_closure = float(np.clip(1.0 - (lip_gap_norm / 0.05), 0.0, 1.0))

    # ---- Jaw vertical position (normalized) ----
    # Distance from forehead to chin — we compare this to a running baseline
    # but for single-frame we just report the face-height-normalized chin Y
    chin = pts[JAW_LANDMARK]
    forehead = pts[FOREHEAD_LANDMARK]
    jaw_drop = float((chin[1] - forehead[1]) / face_height)

    # ---- Facial asymmetry ----
    # For each (left_idx, right_idx) pair, compute horizontal distance from midline
    # A symmetric face has both points equidistant from the midline
    asymmetry_scores = []
    midline_x = (pts[FOREHEAD_LANDMARK][0] + pts[JAW_LANDMARK][0]) / 2.0

    for left_idx, right_idx in SYMMETRY_PAIRS:
        left_dist = abs(pts[left_idx][0] - midline_x)
        right_dist = abs(pts[right_idx][0] - midline_x)
        if max(left_dist, right_dist) > 0:
            diff = abs(left_dist - right_dist) / max(left_dist, right_dist)
            asymmetry_scores.append(diff)

    asymmetry = float(np.mean(asymmetry_scores)) if asymmetry_scores else 0.0

    return {
        "lip_closure": round(lip_closure, 3),
        "lip_gap_normalized": round(float(lip_gap_norm), 4),
        "jaw_drop_ratio": round(jaw_drop, 3),
        "facial_asymmetry": round(asymmetry, 3),
        "face_detected": True,
    }


# ---------------------------------------------------------------
# Base64 → image helper (for browser uploads)
# ---------------------------------------------------------------
def decode_image_base64(data_url: str) -> Optional[np.ndarray]:
    """
    Accept either a full data URL ('data:image/png;base64,...') or just the base64 payload.
    Returns BGR image (OpenCV convention) or None on failure.
    """
    try:
        if "," in data_url:
            _, b64 = data_url.split(",", 1)
        else:
            b64 = data_url

        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[decode_image_base64] error: {e}")
        return None


def analyze_frame_from_base64(data_url: str) -> Optional[dict]:
    img = decode_image_base64(data_url)
    if img is None:
        return None
    return analyze_frame(img)
