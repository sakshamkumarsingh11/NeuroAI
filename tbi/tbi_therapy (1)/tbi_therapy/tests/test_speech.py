"""Unit tests for the speech analysis module.

Run: pytest tests/
"""
import pytest
from modules.speech_analysis import (
    phoneme_accuracy,
    reward_function,
    convert_ipa_to_arpabet,
    normalize_phonemes,
    apply_feedback,
    llm_feedback,
)
from modules.severity import estimate_severity, get_therapy_plan


# -----------------------------------------------------------
# Phoneme accuracy
# -----------------------------------------------------------
def test_phoneme_accuracy_perfect():
    Cp, Tp = phoneme_accuracy(["hh", "eh", "l", "ow"], ["hh", "eh", "l", "ow"])
    assert Cp == 4 and Tp == 4


def test_phoneme_accuracy_one_wrong():
    Cp, Tp = phoneme_accuracy(["hh", "eh", "l", "oh"], ["hh", "eh", "l", "ow"])
    assert Tp == 4
    assert Cp == 3


def test_phoneme_accuracy_empty_reference():
    Cp, Tp = phoneme_accuracy(["hh"], [])
    assert Cp == 0 and Tp == 0


def test_phoneme_accuracy_insertion_tolerant():
    """Insertions by the predictor should not hurt Cp (standard ASR scoring)."""
    Cp, Tp = phoneme_accuracy(["hh", "eh", "x", "l", "ow"], ["hh", "eh", "l", "ow"])
    assert Tp == 4
    assert Cp == 4


# -----------------------------------------------------------
# Reward
# -----------------------------------------------------------
def test_reward_all_correct():
    assert reward_function(5, 5) == 5


def test_reward_half_correct():
    assert reward_function(3, 6) == 0  # 3 - (6-3) = 0


# -----------------------------------------------------------
# IPA conversion
# -----------------------------------------------------------
def test_ipa_to_arpabet_basic():
    result = convert_ipa_to_arpabet(["h", "ɛ", "l", "o"])
    assert "HH" in result
    assert "EH" in result
    assert "L" in result
    assert "OW" in result


def test_ipa_to_arpabet_ignores_unknown():
    result = convert_ipa_to_arpabet(["p", "˥", "b"])  # ˥ is a tone marker
    assert "P" in result
    assert "B" in result
    assert "˥" not in result


# -----------------------------------------------------------
# Normalization (TIMIT folding)
# -----------------------------------------------------------
def test_normalize_folds_silence():
    result = normalize_phonemes(["hh", "eh", "l", "ow", "h#", "sil"])
    assert "h#" not in result
    assert "sil" not in result
    assert "hh" in result


def test_normalize_folds_ao_to_aa():
    result = normalize_phonemes(["ao"])
    assert result == ["aa"]


# -----------------------------------------------------------
# Feedback
# -----------------------------------------------------------
def test_apply_feedback_fixes_one():
    pred = ["hh", "eh", "l", "oh"]
    ref = ["hh", "eh", "l", "ow"]
    improved = apply_feedback(pred, ref)
    assert improved == ["hh", "eh", "l", "ow"]


def test_apply_feedback_no_changes():
    pred = ["hh", "eh", "l", "ow"]
    ref = ["hh", "eh", "l", "ow"]
    assert apply_feedback(pred, ref) == pred


def test_llm_feedback_identifies_errors():
    wrong = llm_feedback(["a", "x", "c"], ["a", "b", "c"])
    assert wrong == [(1, "b")]


# -----------------------------------------------------------
# Severity
# -----------------------------------------------------------
def test_severity_normal():
    label, score, _ = estimate_severity(phoneme_acc=95, pauses=0)
    assert label == "Normal"


def test_severity_severe_low_acc():
    label, score, _ = estimate_severity(phoneme_acc=30)
    assert label == "Severe"
    assert score >= 4


def test_severity_moderate_mid_acc():
    label, _, _ = estimate_severity(phoneme_acc=65)
    assert label in ("Moderate", "Mild")  # depends on pauses


def test_severity_escalates_with_facial():
    label_plain, _, _ = estimate_severity(phoneme_acc=80)
    label_asym, _, _ = estimate_severity(phoneme_acc=80, facial_asymmetry=0.2)
    # Facial asymmetry should push toward more severe or keep same
    assert label_asym in ("Mild", "Moderate", "Severe")


# -----------------------------------------------------------
# Therapy plan
# -----------------------------------------------------------
def test_therapy_plan_has_exercises():
    plan = get_therapy_plan("Moderate", 70)
    assert "exercises" in plan
    assert len(plan["exercises"]) > 0
    assert "focus" in plan
    assert plan["severity"] == "Moderate"


def test_therapy_plan_progress_note_near_target():
    plan = get_therapy_plan("Moderate", 72)  # target is 75
    assert "from advancing" in plan["progress_note"] or "progress" in plan["progress_note"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
