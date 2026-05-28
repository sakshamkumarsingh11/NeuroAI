# TBI Speech Therapy System

AI-powered speech therapy application for Traumatic Brain Injury (TBI) patients, combining:

- **Speech analysis**: Allosaurus phoneme recognition + PPO-based correction
- **Severity estimation**: Clinically-grounded mild/moderate/severe classification
- **RL therapy planner**: Adapts difficulty to patient progress across sessions
- **MediaPipe facial analysis**: Lip closure, jaw movement, facial asymmetry (important for TBI)
- **TTS feedback**: AI therapist speaks to the patient
- **Flask web GUI**: Interactive browser-based interface

## Project Structure

```
tbi_therapy/
├── app.py                      # Flask entry point — run this
├── config.py                   # Paths, thresholds
├── requirements.txt            # pip install -r requirements.txt
├── modules/
│   ├── speech_analysis.py      # Allosaurus + phoneme accuracy
│   ├── severity.py             # Severity estimator + therapy protocols
│   ├── rl_planner.py           # RL difficulty adapter
│   ├── facial_analysis.py      # MediaPipe face mesh
│   ├── tts.py                  # Text-to-speech
│   └── session_manager.py      # SQLite persistence
├── templates/                  # Jinja2 HTML templates
├── static/                     # CSS + JS
├── data/
│   └── cmudict-0.7b            # CMU phoneme dictionary
├── models/                     # saved PPO weights
└── tests/                      # unit tests
```

## Setup (macOS)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CMU dict
curl -o data/cmudict-0.7b https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b

# 4. Initialize database
python -c "from modules.session_manager import init_db; init_db()"

# 5. Run the app
python app.py
```

Open http://localhost:5000 in your browser.

## Permissions on macOS

The app needs microphone and webcam access. Grant these when Safari/Chrome prompts you.
Also: System Settings → Privacy & Security → Microphone / Camera → enable your browser.

## Key differences from Colab version

- No Google Drive paths — everything is local in `data/`
- SQLite instead of in-memory lists (sessions persist)
- MediaPipe integration for video analysis
- Modular — easier to test individual pieces
- Threaded Flask server so audio processing doesn't block the UI

## Running tests

```bash
pytest tests/
```
