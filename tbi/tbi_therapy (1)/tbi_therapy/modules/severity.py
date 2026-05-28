"""TBI severity estimation and therapy protocol recommendation.

Ported from Colab Phase 2 cells.
"""
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------
# Therapy protocols by severity
# ---------------------------------------------------------------
THERAPY_PROTOCOLS = {
    "Severe": {
        "focus": "Vowel production and breathing control",
        "exercises": [
            "Sustained vowel phonation (/a/, /i/, /u/ for 5 seconds each)",
            "Diaphragmatic breathing — 4s inhale, 6s exhale",
            "Lip and tongue range-of-motion warmups",
            "Simple CV syllables: 'pa', 'ma', 'ta', 'ka'",
        ],
        "difficulty": 0,  # numeric for RL
        "difficulty_label": "Single phonemes and CV syllables",
        "session_length_min": 15,
        "frequency_per_week": 5,
        "target_phoneme_acc": 60,
        "word_pool": ["pa", "ma", "ta", "ka", "ba", "da", "ah", "ee", "oo"],
    },
    "Moderate": {
        "focus": "Syllable combinations and common words",
        "exercises": [
            "Multi-syllable words: 'butter', 'coffee', 'mother'",
            "Minimal pairs: pat/bat, sip/zip, think/sink",
            "Stressed vs unstressed syllables",
            "Reading single-word lists aloud with slow rate",
        ],
        "difficulty": 1,
        "difficulty_label": "Words and short phrases",
        "session_length_min": 20,
        "frequency_per_week": 4,
        "target_phoneme_acc": 75,
        "word_pool": ["butter", "coffee", "mother", "water", "little",
                      "happy", "yellow", "garden", "simple", "summer"],
    },
    "Mild": {
        "focus": "Sentence production and prosody",
        "exercises": [
            "Short functional sentences: 'I would like coffee please'",
            "Tongue twisters at comfortable pace",
            "Reading aloud 2–3 sentences with natural intonation",
            "Question-answer pairs about daily life",
        ],
        "difficulty": 2,
        "difficulty_label": "Full sentences and connected speech",
        "session_length_min": 25,
        "frequency_per_week": 3,
        "target_phoneme_acc": 90,
        "word_pool": [
            "The weather is nice today",
            "I would like a cup of tea",
            "She walks to the market every morning",
            "The quick brown fox jumps over the lazy dog",
            "Please pass me the butter",
        ],
    },
    "Normal": {
        "focus": "Conversational maintenance and complex language",
        "exercises": [
            "Spontaneous storytelling from picture prompts",
            "Debate questions (5+ sentence responses)",
            "Oral summarization of a paragraph",
            "Emotional expression and tone variation",
        ],
        "difficulty": 3,
        "difficulty_label": "Spontaneous conversation",
        "session_length_min": 30,
        "frequency_per_week": 2,
        "target_phoneme_acc": None,
        "word_pool": [
            "Describe your favorite holiday memory in detail",
            "What would you do if you won a million dollars",
            "Explain how to make your favorite meal step by step",
        ],
    },
}


# ---------------------------------------------------------------
# Severity estimation
# ---------------------------------------------------------------
def estimate_severity(
    phoneme_acc: float,
    pauses: int = 0,
    audio_duration_sec: float = 3.0,
    facial_asymmetry: float | None = None,
    lip_closure_score: float | None = None,
) -> Tuple[str, int, dict]:
    """
    Estimate TBI severity from speech + (optional) facial features.

    Args:
      phoneme_acc: 0-100
      pauses: silent sample count
      audio_duration_sec: total duration in seconds
      facial_asymmetry: 0=symmetric, 1=very asymmetric (optional)
      lip_closure_score: 0=poor closure, 1=good closure (optional)

    Returns:
      (label, numeric_score, breakdown_dict)
    """
    breakdown = {}
    score = 0

    # --- phoneme accuracy ---
    if phoneme_acc < 40:
        score += 4; breakdown["phoneme"] = "severe_impairment"
    elif phoneme_acc < 60:
        score += 3; breakdown["phoneme"] = "marked_impairment"
    elif phoneme_acc < 75:
        score += 2; breakdown["phoneme"] = "moderate_impairment"
    elif phoneme_acc < 90:
        score += 1; breakdown["phoneme"] = "mild_impairment"
    else:
        breakdown["phoneme"] = "within_normal"

    # --- pause ratio ---
    pause_ratio = 0.0
    if audio_duration_sec > 0:
        total_samples = audio_duration_sec * 16000
        pause_ratio = pauses / total_samples if total_samples > 0 else 0
    breakdown["pause_ratio"] = round(pause_ratio, 3)
    if pause_ratio > 0.5:
        score += 2; breakdown["pauses"] = "excessive"
    elif pause_ratio > 0.3:
        score += 1; breakdown["pauses"] = "elevated"
    else:
        breakdown["pauses"] = "normal"

    # --- facial features (if MediaPipe ran) ---
    if facial_asymmetry is not None:
        breakdown["facial_asymmetry"] = round(facial_asymmetry, 3)
        if facial_asymmetry > 0.15:
            score += 2; breakdown["facial"] = "significant_asymmetry"
        elif facial_asymmetry > 0.08:
            score += 1; breakdown["facial"] = "mild_asymmetry"
        else:
            breakdown["facial"] = "symmetric"

    if lip_closure_score is not None:
        breakdown["lip_closure"] = round(lip_closure_score, 3)
        if lip_closure_score < 0.3:
            score += 2; breakdown["lip"] = "poor_closure"
        elif lip_closure_score < 0.6:
            score += 1; breakdown["lip"] = "reduced_closure"
        else:
            breakdown["lip"] = "good_closure"

    # --- classify ---
    if score >= 5:
        label = "Severe"
    elif score >= 3:
        label = "Moderate"
    elif score >= 1:
        label = "Mild"
    else:
        label = "Normal"

    return label, score, breakdown


# ---------------------------------------------------------------
# Therapy plan retrieval
# ---------------------------------------------------------------
def get_therapy_plan(severity: str, current_accuracy: float) -> dict:
    plan = dict(THERAPY_PROTOCOLS[severity])  # shallow copy
    plan["severity"] = severity
    plan["current_accuracy"] = round(current_accuracy, 2)

    target = plan.get("target_phoneme_acc")
    if target is not None:
        gap = target - current_accuracy
        if gap <= 5:
            plan["progress_note"] = f"Excellent! You're {gap:.1f}% from advancing."
        elif gap <= 15:
            plan["progress_note"] = f"Good progress. {gap:.1f}% to the next level."
        else:
            plan["progress_note"] = "Keep practicing — consistency matters most."
    else:
        plan["progress_note"] = "Maintenance level — great job!"

    return plan


def generate_feedback_text(severity: str, phoneme_acc: float, wrong_phonemes: list) -> str:
    """Generate friendly, encouraging feedback text for the patient."""
    acc = round(phoneme_acc)

    if phoneme_acc >= 90:
        opener = f"Excellent work! You scored {acc} percent."
    elif phoneme_acc >= 70:
        opener = f"Good effort! You got {acc} percent right."
    elif phoneme_acc >= 50:
        opener = f"Nice try. You scored {acc} percent."
    else:
        opener = f"Let's take this slowly. You scored {acc} percent."

    if wrong_phonemes:
        n = len(wrong_phonemes)
        if n == 1:
            detail = f" Just one sound needs a bit more practice."
        else:
            detail = f" {n} sounds need a bit more practice."
    else:
        detail = " Every phoneme was correct!"

    plan = THERAPY_PROTOCOLS[severity]
    closer = f" Next, we'll focus on {plan['focus'].lower()}."

    return opener + detail + closer
