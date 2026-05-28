<div align="center">

<img src="https://img.shields.io/badge/NeuroVoiceAI-TBI%20Rehabilitation%20Platform-c0392b?style=for-the-badge&logo=brain&logoColor=white" alt="NeuroVoiceAI"/>

# NeuroVoiceAI
### AI-Powered Traumatic Brain Injury Rehabilitation Platform

[![Python](https://img.shields.io/badge/Python-3.9--3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**An end-to-end AI ecosystem for early hemorrhage detection and intelligent speech rehabilitation — combining deep learning, reinforcement learning, speech processing, and real-time computer vision.**

[Features](#-features) · [Architecture](#-architecture) · [Tech Stack](#-tech-stack) · [Getting Started](#-getting-started) · [Project Structure](#-project-structure) · [How It Works](#-how-it-works) · [Contributing](#-contributing)

</div>

---

## 🧠 Motivation

Traumatic Brain Injury (TBI) and stroke frequently leave patients with severe motor, speech, and cognitive deficits requiring months of daily rehabilitation. The reality for most patients:

| Problem | Impact |
|---|---|
| 🏥 **Therapist scarcity** | Most patients receive a fraction of the sessions they need |
| 📋 **Subjective tracking** | Progress measured by observation — no objective clinical data |
| 📄 **Static exercises** | Sheets that don't adapt lead to frustration or boredom |
| 🔬 **Slow diagnostics** | Radiologists need fast, explainable CT screening tools |

**NeuroVoiceAI** addresses each of these with an integrated, patient-centric AI ecosystem.

---

## ✨ Features

- 🩻 **CT Scan Hemorrhage Detection** — EfficientNet-B4 classifies 5 sub-types of intracranial hemorrhage with per-class probability scores
- 🔥 **Grad-CAM Explainability** — Heatmap overlays show clinicians *exactly* which brain regions triggered a prediction
- 🎙️ **Phoneme-Level Speech Scoring** — Allosaurus transcribes raw IPA phonemes from dysarthric speech; Levenshtein alignment computes objective pronunciation accuracy
- 🤖 **Adaptive RL Therapist** — A PPO Actor-Critic agent dynamically scales exercise difficulty based on accuracy, fatigue, and recovery trend
- 👄 **Facial Motor Monitoring** — MediaPipe Face Landmarker tracks lip closure, jaw ROM, and facial asymmetry in real time
- 🗣️ **Spoken AI Feedback** — gTTS generates MP3 therapist responses after every exercise attempt
- 📊 **Patient & Doctor Portals** — React dashboards for conducting exercises, reviewing session graphs, and booking consultations
- 🗄️ **Persistent Session Database** — SQLite stores every attempt with full phoneme, facial, and RL metadata

---

## 🏗️ Architecture

The platform is a **modular three-tier application**: one React frontend and two specialised Python backends, each independently deployable.

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend  (Port 3000)                  │
│         Tailwind CSS · Context API · Webcam/Mic Capture         │
└────────────┬───────────────────────────────┬────────────────────┘
             │  CT scan image upload          │  Audio + camera frame
             ▼                                ▼
┌────────────────────────┐      ┌──────────────────────────────────┐
│  FastAPI CT Backend    │      │      Flask Speech Backend        │
│     (Port 8001)        │      │          (Port 5000)             │
│                        │      │                                  │
│  EfficientNet-B4       │      │  Allosaurus  · Phoneme Scoring   │
│  (PyTorch)             │      │  MediaPipe   · Face Landmarker   │
│                        │      │  PPO RL      · Difficulty Agent  │
│  Grad-CAM Heatmaps     │      │  gTTS        · Voice Feedback    │
│  5-class ICH detection │      │  SQLite3     · Session Database  │
└────────────────────────┘      └──────────────────────────────────┘
```

### Data Flow

**CT Scan Pathway**
```
User uploads CT image → FastAPI → EfficientNet-B4 inference
→ Grad-CAM heatmap generation → prediction + overlay returned to frontend
```

**Speech Therapy Pathway**
```
Patient records speech + camera frame → Flask backend
→ Allosaurus phoneme extraction → Levenshtein scoring
→ MediaPipe facial metrics → PPO RL selects next exercise
→ gTTS feedback audio → SQLite persists attempt → results returned
```

---

## 🛠️ Tech Stack

| Tier | Component | Technology | Purpose |
|---|---|---|---|
| **Frontend** | UI & Navigation | React, Tailwind CSS, Context API | Dashboard, charts, webcam/mic capture, patient login |
| **CT Backend** | REST API | FastAPI, Uvicorn | High-performance async server for prediction endpoints |
| **CT Backend** | Classification | PyTorch, torchvision | EfficientNet-B4 fine-tuned for multi-label ICH detection |
| **CT Backend** | Explainability | Grad-CAM, OpenCV | Heatmap overlays pinpointing hemorrhage regions |
| **Speech Backend** | REST API | Flask, CORS | Hosts therapy sessions, DB access, voice triggers |
| **Speech Backend** | Speech Recognition | Allosaurus, Librosa | Transcribes audio → raw IPA phonemes |
| **Speech Backend** | Sequence Scoring | Levenshtein DP | Aligns phonemes to CMU dictionary; scores accuracy |
| **Speech Backend** | Facial Analysis | MediaPipe Face Landmarker | Lip closure, jaw ROM, facial asymmetry detection |
| **Speech Backend** | RL Planner | PyTorch (PPO Actor-Critic) | Adaptive difficulty based on accuracy and fatigue |
| **Speech Backend** | Voice Synthesis | gTTS | Generates spoken MP3 AI therapist feedback |
| **Speech Backend** | Database | SQLite3 | Persists patients, sessions, and attempt records |

---

## ⚡ Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 – 3.11 |
| Node.js | 18+ |
| ffmpeg | Latest (must be in system PATH) |

> **Why ffmpeg?** Librosa requires it to decode `.webm` audio uploaded from modern browsers. Speech analysis will fail silently without it.

---

### Step 1 — CT Scan Backend

```bash
# Navigate to the backend directory
cd tbi/backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Required:** Ensure `best_model.pth` (trained EfficientNet-B4 weights) is present in `./backend/` before starting.

```bash
# Start the FastAPI server
uvicorn app:app --host 127.0.0.1 --port 8001
```

Interactive API docs → `http://127.0.0.1:8001/docs`

---

### Step 2 — Speech Therapy Backend

```bash
# Navigate to the speech backend directory
cd tbi/tbi_therapy\ \(1\)/tbi_therapy

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

Download the CMU Pronouncing Dictionary:

```powershell
# PowerShell
New-Item -ItemType Directory -Force -Path .\data
Invoke-WebRequest `
  -Uri "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b" `
  -OutFile ".\data\cmudict-0.7b"
```

Place the MediaPipe model file in the models directory:

```
tbi_therapy/
└── models/
    └── face_landmarker.task   ← place file here
```

Initialise the SQLite database:

```bash
python -c "from modules.session_manager import init_db; init_db()"
```

Start the Flask server:

```bash
python app.py
# Server running at http://127.0.0.1:5000
```

---

### Step 3 — React Frontend

```bash
# Navigate to the frontend directory
cd tbi/frontend

# Install packages
npm install
```

Configure environment variables — create a `.env` file:

```env
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_VOICE_API_URL=http://localhost:5000
REACT_APP_CT_SCAN_API_URL=http://localhost:8001
```

```bash
# Start the development server
npm start
# Open http://localhost:3000
```

> Grant **microphone and camera permissions** when the browser prompts — both are required for speech scoring and facial analysis.

---

### Port Summary

| Service | Port | URL |
|---|---|---|
| React Frontend | 3000 | http://localhost:3000 |
| CT Scan Backend (FastAPI) | 8001 | http://127.0.0.1:8001 |
| Speech Backend (Flask) | 5000 | http://127.0.0.1:5000 |

---

## 📁 Project Structure

```
NeuroVoiceAI/
└── tbi/
    ├── backend/                         # CT Scan Analysis Backend (FastAPI)
    │   ├── app.py                       # FastAPI entry point & routes
    │   ├── model.py                     # EfficientNet-B4 architecture definition
    │   ├── predict.py                   # Inference logic & decision thresholds
    │   ├── gradcam.py                   # Grad-CAM heatmap generation
    │   ├── best_model.pth               # Trained model weights (not tracked by git)
    │   └── requirements.txt
    │
    ├── tbi_therapy (1)/
    │   └── tbi_therapy/                 # Speech Therapy Backend (Flask)
    │       ├── app.py                   # Flask entry point & API routes
    │       ├── modules/
    │       │   ├── speech_analysis.py   # Allosaurus + phoneme scoring pipeline
    │       │   ├── rl_planner.py        # PPO Actor-Critic RL agent
    │       │   ├── facial_analysis.py   # MediaPipe facial feature extraction
    │       │   ├── session_manager.py   # SQLite database schema & queries
    │       │   └── severity.py          # Therapy protocols by difficulty level
    │       ├── models/
    │       │   ├── face_landmarker.task # MediaPipe model file
    │       │   ├── ppo_actor.pt         # RL actor weights (auto-saved)
    │       │   └── ppo_critic.pt        # RL critic weights (auto-saved)
    │       ├── data/
    │       │   └── cmudict-0.7b         # CMU Pronouncing Dictionary
    │       └── requirements.txt
    │
    └── frontend/                        # React Frontend
        ├── src/
        │   ├── pages/
        │   │   ├── CTScanPage.jsx       # CT scan upload & results view
        │   │   └── VoiceAssistantPage.jsx # Speech therapy session UI
        │   └── context/                 # React Context API (auth, patient state)
        ├── .env                         # Backend URL configuration
        └── package.json
```

---

## 🔬 How It Works

### CT Scan Analysis

The backend uses **EfficientNet-B4** fine-tuned for multi-label intracranial hemorrhage classification. It detects 5 sub-types simultaneously, since TBI patients often present with multiple concurrent bleeds:

| Code | Type | Location |
|---|---|---|
| **EDH** | Epidural | Between skull and dura mater |
| **IPH** | Intraparenchymal | Inside brain tissue |
| **IVH** | Intraventricular | Inside fluid-filled brain chambers |
| **SAH** | Subarachnoid | Beneath the arachnoid membrane |
| **SDH** | Subdural | Between dura and arachnoid membrane |

The classifier head is: `Dropout(0.4) → Linear(512) → BatchNorm → ReLU → Dropout(0.3) → Linear(5)`, with sigmoid outputs and per-class thresholds `[0.45, 0.5, 0.5, 0.45, 0.45]`.

**Grad-CAM** generates clinician-interpretable heatmaps by hooking the final convolutional block, computing gradient-weighted activation maps, and overlaying them on the original scan (red = high model attention, blue = low attention).

---

### Speech Therapy Pipeline

```
Patient Audio
     │
     ▼
1. Allosaurus ──────► Raw IPA phonemes  e.g. [ɹ, e, d]
     │
     ▼
2. IPA → ARPAbet ───► e.g. [R, EH, D]
     │
     ▼
3. TIMIT Folding ───► 61 classes → 39 (dialect normalisation)
     │
     ▼
4. Levenshtein DP ──► Align vs. CMU dict reference
     │
     ▼
Accuracy = (Correctly aligned phonemes / Total reference phonemes) × 100%
```

**Why Allosaurus?** Standard ASR engines fail for dysarthric (slurred, distorted) speech common in TBI patients. Allosaurus is an unsupervised multilingual model extracting raw phonemes without vocabulary matching — making it uniquely suitable for impaired speech.

---

### Reinforcement Learning Planner

The PPO Agent acts as a live, adaptive therapist:

**State vector (5 features):** Severity level · Recent accuracy · Accuracy trend · Session count · Fatigue proxy (pause ratio)

**Actions:** `0 = Easier` · `1 = Same` · `2 = Harder`

**Reward shaping:**
- `+reward` for pronunciation improvement
- `+1.0` bonus for sustaining high accuracy after advancing
- `−1.5` penalty if accuracy drops below 40% after difficulty increase
- `−0.5` penalty for early session exit

**Therapy levels:**

| Level | Severity | Exercise Type | Example |
|---|---|---|---|
| 0 | Severe | Sustained vowels & CV syllables | `/a/`, `/i/`, `pa`, `ma` |
| 1 | Moderate | Multi-syllables & minimal pairs | `butter`, `pat/bat` |
| 2 | Mild | Conversational sentences | *"I would like a cup of tea"* |
| 3 | Normal | Storytelling & open discourse | Free-form prompts |

RL model weights persist across sessions — actor/critic networks auto-save to `./models/ppo_actor.pt` and `./models/ppo_critic.pt` after every update.

---

### Database Schema

Three tables in a 1-to-many hierarchy:

```
PATIENTS (1) ──► SESSIONS (*) ──► ATTEMPTS (*)
```

Each `ATTEMPT` record stores: target text, predicted/reference phonemes, phoneme accuracy, speech rate, pause ratio, fluency, duration, lip closure, facial asymmetry, jaw drop, RL action taken, and next difficulty — enabling rich longitudinal analysis.

---

## 🔧 Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Speech analysis returns an error | `ffmpeg` not found in PATH | Install ffmpeg and add it to system PATH |
| CT backend fails to start | `best_model.pth` missing | Add trained EfficientNet weights to `./backend/` |
| MediaPipe errors on startup | `face_landmarker.task` missing | Download model and place in `./models/` |
| Frontend can't reach backend | Wrong `.env` ports | Verify `.env` matches the ports each backend is running on |
| Browser blocks mic/camera | Permissions denied | Reload page and allow both permissions when prompted |

---

## 🤝 Contributing

Contributions are welcome! To add new therapy words or difficulty tiers, modify the `THERAPY_PROTOCOLS` dictionaries in `modules/severity.py` — word lists are dynamically selected based on the active RL difficulty level.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👥 Team

Built with ❤️ as a group project at the intersection of deep learning, reinforcement learning, speech processing, and clinical rehabilitation technology.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you found this useful, please star the repository!**

[![GitHub stars](https://img.shields.io/github/stars/sakshamkumarsingh11/NeuroVoiceAI?style=social)](https://github.com/sakshamkumarsingh11/NeuroVoiceAI)

</div>
