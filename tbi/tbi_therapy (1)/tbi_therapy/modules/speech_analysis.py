"""Speech analysis: phoneme prediction, accuracy, feedback.

Ported from Colab notebook — consolidates cells 1, 3, 17, 31–35, 38, 42–43.
"""
import re
import string
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from allosaurus.app import read_recognizer

from config import CMU_DICT_PATH, SAMPLE_RATE

# -------------------------------------------------------------------
# CMU dictionary loading
# -------------------------------------------------------------------
_cmu_dict: dict | None = None


def load_cmu_dict() -> dict:
    """Lazy-load the CMU phoneme dictionary."""
    global _cmu_dict
    if _cmu_dict is not None:
        return _cmu_dict

    if not CMU_DICT_PATH.exists():
        raise FileNotFoundError(
            f"CMU dict not found at {CMU_DICT_PATH}. "
            "Download with: curl -o data/cmudict-0.7b "
            "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
        )

    d = {}
    with open(CMU_DICT_PATH, "r", encoding="latin-1") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.strip().split("  ")
            if len(parts) < 2:
                continue
            word = parts[0].lower().split("(")[0]
            phonemes = parts[1].split()
            d.setdefault(word, []).append(phonemes)
    _cmu_dict = d
    return d


def text_to_phonemes(sentence: str) -> List[str]:
    """Convert text to CMU phonemes."""
    cmu = load_cmu_dict()
    phonemes = []
    for w in sentence.lower().split():
        w = w.strip(string.punctuation)
        if w in cmu:
            phonemes.extend(cmu[w][0])
    return phonemes


# -------------------------------------------------------------------
# Allosaurus (lazy-loaded singleton)
# -------------------------------------------------------------------
_allo_model = None


def get_allosaurus():
    global _allo_model
    if _allo_model is None:
        _allo_model = read_recognizer()
    return _allo_model


# -------------------------------------------------------------------
# Phoneme mappings
# -------------------------------------------------------------------
IPA_TO_ARPABET = {
    # vowels
    'i': 'IY', 'ɪ': 'IH', 'e': 'EY', 'ɛ': 'EH', 'æ': 'AE',
    'ɑ': 'AA', 'ɒ': 'AA', 'ʌ': 'AH', 'ə': 'AH', 'ɚ': 'ER', 'ɝ': 'ER',
    'ɔ': 'AO', 'o': 'OW', 'ʊ': 'UH', 'u': 'UW',
    'aɪ': 'AY', 'aʊ': 'AW', 'ɔɪ': 'OY',
    # stops
    'p': 'P', 'b': 'B', 't': 'T', 'd': 'D', 'k': 'K', 'g': 'G', 'ɡ': 'G',
    # affricates
    'tʃ': 'CH', 'dʒ': 'JH',
    # fricatives
    'f': 'F', 'v': 'V', 'θ': 'TH', 'ð': 'DH',
    's': 'S', 'z': 'Z', 'ʃ': 'SH', 'ʒ': 'ZH', 'h': 'HH',
    # nasals
    'm': 'M', 'n': 'N', 'ŋ': 'NG',
    # liquids / glides
    'l': 'L', 'ɹ': 'R', 'r': 'R', 'j': 'Y', 'w': 'W',
}

# TIMIT 61 → 39 folding (Lee & Hon 1989)
PHONEME_MAP = {
    "aa": "aa", "ao": "aa", "ae": "ae",
    "ah": "ah", "ax": "ah", "ax-h": "ah",
    "aw": "aw", "ay": "ay", "eh": "eh",
    "er": "er", "axr": "er", "ey": "ey",
    "ih": "ih", "ix": "ih", "iy": "iy",
    "ow": "ow", "oy": "oy", "uh": "uh",
    "uw": "uw", "ux": "uw",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g",
    "ch": "ch", "jh": "jh",
    "f": "f", "v": "v", "th": "th", "dh": "dh",
    "s": "s", "z": "z", "sh": "sh", "zh": "sh",
    "hh": "hh", "hv": "hh",
    "m": "m", "em": "m", "n": "n", "nx": "n", "en": "n",
    "ng": "ng", "eng": "ng",
    "l": "l", "el": "l", "r": "r", "w": "w", "y": "y",
    "sil": "", "sp": "", "pau": "", "epi": "", "h#": "", "#h": "",
    "bcl": "", "dcl": "", "gcl": "", "kcl": "", "pcl": "", "tcl": "", "q": "",
    "dx": "t",
}


def convert_ipa_to_arpabet(pred: List[str]) -> List[str]:
    out = []
    for p in pred:
        p = p.strip()
        if p in IPA_TO_ARPABET:
            out.append(IPA_TO_ARPABET[p])
        elif len(p) > 1 and p[0] in IPA_TO_ARPABET:
            out.append(IPA_TO_ARPABET[p[0]])
    return out


def normalize_phonemes(ph_list: List[str]) -> List[str]:
    out = []
    for p in ph_list:
        p = p.lower()
        if p in PHONEME_MAP:
            mapped = PHONEME_MAP[p]
            if mapped:
                out.append(mapped)
    return out


# -------------------------------------------------------------------
# Audio preprocessing + prediction
# -------------------------------------------------------------------
def preprocess_audio(input_path: str, output_path: str = "temp.wav") -> str | None:
    try:
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)
        sf.write(output_path, audio, sr)
        return output_path
    except Exception as e:
        print(f"[preprocess_audio] error: {e}")
        return None


def predict_phonemes(audio_path: str) -> List[str]:
    """Run Allosaurus on an audio file and return IPA phonemes."""
    try:
        clean = preprocess_audio(audio_path)
        if clean is None:
            return []
        output = get_allosaurus().recognize(clean)
        return output.split()
    except Exception as e:
        print(f"[predict_phonemes] error: {e}")
        return []


def predict_and_normalize(audio_path: str) -> List[str]:
    """Full pipeline: audio → IPA → ARPAbet → folded ARPAbet."""
    ipa = predict_phonemes(audio_path)
    if not ipa:
        return []
    arpabet = convert_ipa_to_arpabet(ipa)
    return normalize_phonemes([p.lower() for p in arpabet])


# -------------------------------------------------------------------
# Accuracy + reward
# -------------------------------------------------------------------
def phoneme_accuracy(predicted: List[str], reference: List[str]) -> Tuple[int, int]:
    """Edit-distance based phoneme accuracy (Lee-Hon folded).

    Returns (Cp, Tp) where Cp = correctly aligned phonemes, Tp = reference length.
    Insertions by the predictor are NOT counted against Cp — standard ASR scoring.
    """
    m, n = len(predicted), len(reference)
    if n == 0:
        return 0, 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if predicted[i - 1] == reference[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)

    # Backtrack to separate substitutions from deletions
    i, j = m, n
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and predicted[i - 1] == reference[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            D += 1; j -= 1
        else:
            I += 1; i -= 1

    Cp = n - S - D
    return Cp, n


def reward_function(Cp: int, Tp: int) -> int:
    """Paper eq. (4): R = Cp - (Tp - Cp)."""
    return Cp - (Tp - Cp)


# -------------------------------------------------------------------
# Audio features (for severity)
# -------------------------------------------------------------------
def extract_features(audio_path: str) -> Tuple[float, float, int, float]:
    """Returns (fluency, speech_rate_proxy, silent_samples, duration_sec)."""
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        fluency = float(np.mean(audio ** 2))
        duration = librosa.get_duration(y=audio, sr=sr)
        speech_rate = len(audio) / (duration + 1e-5)
        pauses = int(np.sum(np.abs(audio) < 0.01))
        return fluency, speech_rate, pauses, duration
    except Exception as e:
        print(f"[extract_features] error: {e}")
        return 0.0, 0.0, 0, 0.0


# -------------------------------------------------------------------
# LLM-simulation feedback (from Colab cell 17)
# -------------------------------------------------------------------
def llm_feedback(predicted: List[str], reference: List[str]) -> List[Tuple[int, str]]:
    """Identify wrong phonemes, return list of (index, correct_phoneme)."""
    return [(i, reference[i])
            for i in range(min(len(predicted), len(reference)))
            if predicted[i] != reference[i]]


def apply_feedback(predicted: List[str], reference: List[str]) -> List[str]:
    """Apply one-phoneme correction per call (simulates LLM guidance)."""
    improved = predicted.copy()
    for i in range(min(len(predicted), len(reference))):
        if predicted[i] != reference[i]:
            improved[i] = reference[i]
            break
    return improved


# -------------------------------------------------------------------
# High-level API used by Flask routes
# -------------------------------------------------------------------
def analyze_utterance(audio_path: str, target_text: str) -> dict:
    """
    End-to-end analysis of one utterance.

    Returns a dict with all metrics needed for severity + therapy planning.
    """
    ref = text_to_phonemes(target_text)
    if not ref:
        return {"error": "Could not phonemize target text"}

    pred = predict_and_normalize(audio_path)
    ref_norm = normalize_phonemes([p.lower() for p in ref])

    if not pred:
        return {"error": "Could not analyze audio — empty prediction"}

    # Apply LLM-simulation feedback step
    pred_corrected = apply_feedback(pred, ref_norm)

    Cp, Tp = phoneme_accuracy(pred_corrected, ref_norm)
    phoneme_acc = (Cp / Tp * 100) if Tp > 0 else 0

    fluency, speech_rate, pauses, duration = extract_features(audio_path)

    return {
        "phoneme_acc": round(phoneme_acc, 2),
        "Cp": Cp,
        "Tp": Tp,
        "predicted_phonemes": pred,
        "reference_phonemes": ref_norm,
        "corrected_phonemes": pred_corrected,
        "wrong_phonemes": llm_feedback(pred, ref_norm),
        "fluency": round(fluency, 6),
        "speech_rate": round(speech_rate, 2),
        "pauses": pauses,
        "duration_sec": round(duration, 2),
        "target_text": target_text,
    }
