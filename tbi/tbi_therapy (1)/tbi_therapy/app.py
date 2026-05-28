"""Flask app — pure JSON API backend for TBI Speech Therapy.

Run: python app.py
The React frontend connects to this service via REACT_APP_VOICE_API_URL.
"""
import os
import random
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from config import AUDIO_UPLOAD_DIR, FLASK_CONFIG
from modules.facial_analysis import analyze_frame_from_base64
from modules.rl_planner import get_planner
from modules.session_manager import (create_patient, end_session,
                                      get_patient,
                                      get_recent_accuracy_history,
                                      get_session_attempts, init_db,
                                      list_patients, list_sessions,
                                      record_attempt,
                                      session_count_for_patient,
                                      start_session)
from modules.severity import (THERAPY_PROTOCOLS, estimate_severity,
                               generate_feedback_text, get_therapy_plan)
from modules.speech_analysis import analyze_utterance
from modules.tts import speak_to_file

# Disable static/template serving — this is a pure API service
app = Flask(__name__, static_folder=None, template_folder=None)
app.config.update(FLASK_CONFIG)
CORS(app)

# Initialize DB on startup
init_db()


# ===============================================================
# Health check
# ===============================================================
@app.route("/")
def health():
    return jsonify({"status": "ok", "service": "tbi_therapy"})


# ===============================================================
# API: patients
# ===============================================================
@app.post("/api/patients")
def api_create_patient():
    data = request.json or {}
    pid = create_patient(
        name=data.get("name", "Anonymous"),
        age=data.get("age"),
        tbi_date=data.get("tbi_date"),
        notes=data.get("notes", ""),
    )
    return jsonify({"id": pid})


# ===============================================================
# API: therapy session
# ===============================================================
@app.post("/api/session/start")
def api_start_session():
    data = request.json or {}
    patient_id = data["patient_id"]
    session_id = start_session(patient_id)

    # Pick a starting word — based on patient's most recent severity
    history = get_recent_accuracy_history(patient_id, n=3)
    if history:
        avg = sum(h["phoneme_acc"] for h in history) / len(history)
        sev, _, _ = estimate_severity(avg)
    else:
        sev = "Moderate"  # sensible default for new patients

    word = random.choice(THERAPY_PROTOCOLS[sev]["word_pool"])

    return jsonify({
        "session_id": session_id,
        "target_word": word,
        "severity": sev,
        "focus": THERAPY_PROTOCOLS[sev]["focus"],
    })


@app.post("/api/session/<int:session_id>/attempt")
def api_attempt(session_id: int):
    """
    Handle one therapy attempt:
      - Save uploaded audio
      - Run speech analysis
      - Run facial analysis (if image provided)
      - Compute severity
      - Use RL to pick next difficulty
      - Record attempt
      - Return feedback
    """
    target_text = request.form.get("target_text", "").strip()
    patient_id = int(request.form.get("patient_id", 0))

    if not target_text:
        return jsonify({"error": "Missing target_text"}), 400

    # ----- 1. Save audio -----
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_path = AUDIO_UPLOAD_DIR / f"{uuid.uuid4().hex}.webm"
    audio_file.save(audio_path)

    # Convert to wav (librosa can read webm directly thanks to audioread/ffmpeg)
    # Our preprocess_audio inside analyze_utterance re-samples to 16kHz wav.

    # ----- 2. Speech analysis -----
    analysis = analyze_utterance(str(audio_path), target_text)
    if "error" in analysis:
        return jsonify(analysis), 400

    # ----- 3. Facial analysis (optional) -----
    facial_info = None
    face_image = request.form.get("face_image_b64")
    if face_image:
        facial_info = analyze_frame_from_base64(face_image)

    # ----- 4. Severity -----
    lip_closure = facial_info["lip_closure"] if facial_info else None
    asymmetry = facial_info["facial_asymmetry"] if facial_info else None

    severity_label, severity_score, breakdown = estimate_severity(
        phoneme_acc=analysis["phoneme_acc"],
        pauses=analysis["pauses"],
        audio_duration_sec=analysis["duration_sec"],
        facial_asymmetry=asymmetry,
        lip_closure_score=lip_closure,
    )

    severity_info = {
        "severity": severity_label,
        "score": severity_score,
        "breakdown": breakdown,
    }

    # ----- 5. RL planner -----
    planner = get_planner()
    history = get_recent_accuracy_history(patient_id, n=5) if patient_id else []
    recent_acc = analysis["phoneme_acc"]
    previous_acc = history[1]["phoneme_acc"] if len(history) > 1 else recent_acc
    trend = recent_acc - previous_acc

    state = planner.encode_state(
        severity=severity_label,
        recent_accuracy=recent_acc,
        accuracy_trend=trend,
        session_count=session_count_for_patient(patient_id) if patient_id else 0,
        fatigue=breakdown["pause_ratio"],
    )

    action_idx, action_label, action_prob = planner.select_action(state)
    next_difficulty = planner.recommend_difficulty(severity_label, action_label)
    next_word = random.choice(THERAPY_PROTOCOLS[next_difficulty]["word_pool"])

    # Compute reward for training (compare to previous attempt)
    reward = planner.compute_reward(
        accuracy_before=previous_acc,
        accuracy_after=recent_acc,
        action=action_label,
    )

    # One-step PPO update (online learning)
    planner.update([{
        "state": state,
        "action_idx": action_idx,
        "old_prob": action_prob,
        "reward": reward,
        "next_state": None,
    }])

    # ----- 6. Generate feedback text + TTS -----
    feedback_text = generate_feedback_text(
        severity_label, recent_acc, analysis["wrong_phonemes"]
    )
    feedback_audio_path = speak_to_file(feedback_text)

    # ----- 7. Record attempt -----
    record_attempt(
        session_id=session_id,
        target_text=target_text,
        analysis=analysis,
        severity_info=severity_info,
        facial_info=facial_info,
        rl_action=action_label,
        rl_next=next_difficulty,
    )

    # ----- 8. Respond -----
    return jsonify({
        "analysis": analysis,
        "severity": severity_info,
        "facial": facial_info,
        "rl": {
            "action": action_label,
            "action_prob": round(action_prob, 3),
            "next_difficulty": next_difficulty,
            "reward": round(reward, 3),
        },
        "next_word": next_word,
        "feedback_text": feedback_text,
        "feedback_audio_url": f"/api/audio/{Path(feedback_audio_path).name}",
        "therapy_plan": get_therapy_plan(severity_label, recent_acc),
    })


@app.post("/api/session/<int:session_id>/end")
def api_end_session(session_id: int):
    attempts = get_session_attempts(session_id)
    if not attempts:
        end_session(session_id, 0, "Unknown", "Empty session")
        return jsonify({"ok": True, "summary": "No attempts recorded."})

    accs = [a["phoneme_acc"] for a in attempts if a["phoneme_acc"] is not None]
    avg_acc = sum(accs) / len(accs) if accs else 0

    severities = [a["severity"] for a in attempts if a["severity"]]
    dominant = max(set(severities), key=severities.count) if severities else "Unknown"

    end_session(session_id, avg_acc, dominant)

    return jsonify({
        "ok": True,
        "summary": {
            "attempts": len(attempts),
            "average_accuracy": round(avg_acc, 2),
            "dominant_severity": dominant,
        },
    })


# ===============================================================
# API: TTS audio file
# ===============================================================
@app.get("/api/audio/<path:filename>")
def api_audio_file(filename: str):
    from config import DATA_DIR
    path = DATA_DIR / "tts_cache" / filename
    if not path.exists():
        return "Not found", 404
    return send_file(path, mimetype="audio/mpeg")


# ===============================================================
# API: speak arbitrary text (for "hear the word" button)
# ===============================================================
@app.post("/api/speak")
def api_speak():
    data = request.json or {}
    text = data.get("text", "").strip()
    slow = bool(data.get("slow", False))
    if not text:
        return jsonify({"error": "No text"}), 400
    path = speak_to_file(text, slow=slow)
    return jsonify({"audio_url": f"/api/audio/{Path(path).name}"})


# ===============================================================
# API: MediaPipe on a single frame
# ===============================================================
@app.post("/api/face/analyze")
def api_face_analyze():
    data = request.json or {}
    img_b64 = data.get("image_b64")
    if not img_b64:
        return jsonify({"error": "No image"}), 400
    result = analyze_frame_from_base64(img_b64)
    if result is None:
        return jsonify({"face_detected": False})
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
