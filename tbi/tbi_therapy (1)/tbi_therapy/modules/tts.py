"""Text-to-speech for AI therapist voice output."""
import hashlib
from pathlib import Path

from gtts import gTTS

from config import DATA_DIR

TTS_CACHE_DIR = DATA_DIR / "tts_cache"
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(text: str, slow: bool) -> Path:
    key = hashlib.md5(f"{text}|{slow}".encode()).hexdigest()[:16]
    return TTS_CACHE_DIR / f"tts_{key}.mp3"


def speak_to_file(text: str, slow: bool = False, lang: str = "en") -> str:
    """
    Generate TTS audio. Returns path to mp3 file.

    Cached by (text, slow) so repeated prompts don't re-hit the network.
    """
    path = _cache_path(text, slow)
    if path.exists():
        return str(path)

    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(str(path))
    return str(path)


def clear_cache():
    for f in TTS_CACHE_DIR.glob("tts_*.mp3"):
        f.unlink()
