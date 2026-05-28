// ============================================================
// TBI Therapy — Frontend logic
// Handles: webcam, mic recording, MediaPipe polling, API calls
// ============================================================

const state = {
  patientId: null,
  sessionId: null,
  targetWord: "",
  mediaRecorder: null,
  audioChunks: [],
  recording: false,
  stream: null,
  facePollInterval: null,
  lastFacial: null,
};

// ============================================================
// Boot
// ============================================================
window.addEventListener("DOMContentLoaded", async () => {
  const root = document.querySelector(".therapy-layout");
  state.patientId = parseInt(root.dataset.patientId, 10);

  try {
    await setupWebcam();
  } catch (err) {
    console.warn("Webcam not available:", err);
    document.getElementById("m-face").textContent = "no camera";
  }

  await startSession();
  startFacialPolling();
});

// ============================================================
// Session
// ============================================================
async function startSession() {
  const res = await fetch("/api/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_id: state.patientId }),
  });
  const data = await res.json();
  state.sessionId = data.session_id;
  state.targetWord = data.target_word;
  document.getElementById("target-word").textContent = data.target_word;
  setFeedback(`Starting at <strong>${data.severity}</strong> level — focus: ${data.focus}`);
}

async function endSession() {
  stopFacialPolling();
  if (state.stream) state.stream.getTracks().forEach(t => t.stop());

  const res = await fetch(`/api/session/${state.sessionId}/end`, { method: "POST" });
  const data = await res.json();

  const modal = document.getElementById("summary-modal");
  modal.classList.remove("hidden");
  const s = data.summary || {};
  document.getElementById("summary-content").innerHTML = `
    <p><strong>Attempts:</strong> ${s.attempts ?? 0}</p>
    <p><strong>Average accuracy:</strong> ${s.average_accuracy ?? "—"}%</p>
    <p><strong>Dominant severity:</strong> ${s.dominant_severity ?? "—"}</p>
  `;
}

// ============================================================
// Webcam setup
// ============================================================
async function setupWebcam() {
  state.stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: true,
  });
  const video = document.getElementById("webcam");
  video.srcObject = state.stream;
  await video.play();
}

// ============================================================
// MediaPipe polling (send a frame every 2s while recording)
// ============================================================
function startFacialPolling() {
  state.facePollInterval = setInterval(captureAndAnalyzeFrame, 2000);
}
function stopFacialPolling() {
  if (state.facePollInterval) clearInterval(state.facePollInterval);
}

async function captureAndAnalyzeFrame() {
  const video = document.getElementById("webcam");
  if (!video.videoWidth) return;

  const canvas = document.getElementById("face-canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  const dataUrl = canvas.toDataURL("image/jpeg", 0.7);

  try {
    const res = await fetch("/api/face/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_b64: dataUrl }),
    });
    const data = await res.json();

    if (data.face_detected === false) {
      document.getElementById("m-face").textContent = "none";
      document.getElementById("m-lip").textContent = "—";
      document.getElementById("m-sym").textContent = "—";
      state.lastFacial = null;
      return;
    }

    document.getElementById("m-face").textContent = "detected ✓";
    document.getElementById("m-lip").textContent = data.lip_closure.toFixed(2);
    document.getElementById("m-sym").textContent = data.facial_asymmetry.toFixed(3);
    state.lastFacial = data;
    // store the base64 so we can send the actual frame with the audio
    state.lastFaceImage = dataUrl;
  } catch (err) {
    console.error("Face analyze failed:", err);
  }
}

// ============================================================
// Hear target word (TTS)
// ============================================================
async function playTargetWord(slow = false) {
  const res = await fetch("/api/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: state.targetWord, slow }),
  });
  const data = await res.json();
  if (data.audio_url) {
    const audio = new Audio(data.audio_url);
    audio.play();
  }
}

// ============================================================
// Recording
// ============================================================
async function toggleRecord() {
  if (state.recording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  if (!state.stream) {
    alert("Microphone not available. Please grant permission and reload.");
    return;
  }

  const audioTrack = state.stream.getAudioTracks()[0];
  if (!audioTrack) {
    alert("No audio track found.");
    return;
  }

  state.audioChunks = [];
  const audioStream = new MediaStream([audioTrack]);
  state.mediaRecorder = new MediaRecorder(audioStream, { mimeType: "audio/webm" });

  state.mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) state.audioChunks.push(e.data);
  };

  state.mediaRecorder.onstop = async () => {
    const blob = new Blob(state.audioChunks, { type: "audio/webm" });
    await submitAttempt(blob);
  };

  state.mediaRecorder.start();
  state.recording = true;
  const btn = document.getElementById("btn-record");
  btn.textContent = "⏹ Stop";
  btn.classList.add("recording");
  document.getElementById("record-status").textContent = "Recording…";
}

function stopRecording() {
  if (state.mediaRecorder && state.recording) {
    state.mediaRecorder.stop();
    state.recording = false;
    const btn = document.getElementById("btn-record");
    btn.textContent = "🎤 Record";
    btn.classList.remove("recording");
    document.getElementById("record-status").textContent = "Analyzing…";
  }
}

// ============================================================
// Submit attempt to backend
// ============================================================
async function submitAttempt(audioBlob) {
  const form = new FormData();
  form.append("audio", audioBlob, "attempt.webm");
  form.append("target_text", state.targetWord);
  form.append("patient_id", state.patientId);
  if (state.lastFaceImage) form.append("face_image_b64", state.lastFaceImage);

  try {
    const res = await fetch(`/api/session/${state.sessionId}/attempt`, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Server error" }));
      setFeedback(`<span style="color:var(--danger)">Error: ${err.error || res.status}</span>`);
      document.getElementById("record-status").textContent = "Error — try again";
      return;
    }

    const data = await res.json();
    renderAnalysis(data);
    playFeedbackAudio(data.feedback_audio_url);
    document.getElementById("record-status").textContent = "Done";
    document.getElementById("btn-next").disabled = false;
  } catch (err) {
    console.error(err);
    setFeedback(`<span style="color:var(--danger)">Network error: ${err.message}</span>`);
  }
}

// ============================================================
// Render analysis results
// ============================================================
function renderAnalysis(data) {
  const a = data.analysis;
  const sev = data.severity;
  const rl = data.rl;

  // Accuracy bar
  const acc = a.phoneme_acc;
  document.getElementById("acc-bar").style.width = `${Math.min(acc, 100)}%`;
  document.getElementById("acc-text").textContent = `${acc.toFixed(1)}%`;

  // Severity
  const sevEl = document.getElementById("sev-text");
  sevEl.textContent = sev.severity;
  const badge = document.getElementById("sev-badge");
  badge.textContent = sev.severity;
  badge.className = `badge badge-${sev.severity.toLowerCase()}`;

  // RL
  document.getElementById("rl-text").textContent = `${rl.action} → ${rl.next_difficulty}`;

  // Phonemes
  document.getElementById("pred-phonemes").textContent = a.predicted_phonemes.join(" ") || "—";
  document.getElementById("ref-phonemes").textContent = a.reference_phonemes.join(" ") || "—";

  // Feedback
  setFeedback(data.feedback_text);

  // Cache next word
  state.nextWord = data.next_word;
}

function setFeedback(html) {
  document.getElementById("feedback-box").innerHTML = html;
}

function playFeedbackAudio(url) {
  if (!url) return;
  const audio = new Audio(url);
  audio.play().catch(err => console.warn("Autoplay blocked:", err));
}

// ============================================================
// Next word
// ============================================================
async function nextWord() {
  if (state.nextWord) {
    state.targetWord = state.nextWord;
    document.getElementById("target-word").textContent = state.nextWord;
    document.getElementById("btn-next").disabled = true;
    document.getElementById("record-status").textContent = "Idle";
    setFeedback("<em>Ready for next attempt…</em>");

    // Reset displays
    document.getElementById("acc-bar").style.width = "0%";
    document.getElementById("acc-text").textContent = "—";
    document.getElementById("pred-phonemes").textContent = "—";
    document.getElementById("ref-phonemes").textContent = "—";
  }
}

// Expose for inline onclick handlers
window.togglePatientForm = () => {};  // stub
window.createPatient = () => {};       // stub
window.toggleRecord = toggleRecord;
window.playTargetWord = playTargetWord;
window.nextWord = nextWord;
window.endSession = endSession;
