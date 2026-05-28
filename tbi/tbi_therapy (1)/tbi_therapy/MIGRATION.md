# Migration checklist: Colab → VS Code

## Step 1: Set up the project folder

```bash
# Pick a spot for the project
cd ~/Documents
# Unzip or move the tbi_therapy/ folder here
cd tbi_therapy
```

## Step 2: Create a virtual environment (macOS)

```bash
python3 -m venv venv
source venv/bin/activate

# Confirm you're in the venv
which python   # should show .../tbi_therapy/venv/bin/python
```

## Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS-specific notes:**
- If `mediapipe` fails: `pip install mediapipe==0.10.14 --no-cache-dir`
- If `soundfile` complains about libsndfile: `brew install libsndfile`
- If `librosa` fails on M1/M2 Mac: `pip install librosa --no-binary :all:`
- If `openai-whisper` fails: make sure you have ffmpeg installed (`brew install ffmpeg`)

## Step 4: Download the CMU dictionary

```bash
curl -o data/cmudict-0.7b https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b
```

## Step 5: Initialize the database

```bash
python -c "from modules.session_manager import init_db; init_db()"
```

You should see `[session_manager] initialized DB at .../instance/therapy.db`.

## Step 6: Verify with tests

```bash
pytest tests/ -v
```

All tests should pass. If any fail, that tells you which module has an issue.

## Step 7: Run the app

```bash
python app.py
```

Open `http://localhost:5000` in Chrome or Safari.

### macOS permissions you'll need to grant

When you click "Start therapy session" and then "Record", macOS will prompt for:

1. **Microphone access** — grant for Chrome/Safari in `System Settings → Privacy & Security → Microphone`
2. **Camera access** — same menu → Camera
3. **Screen recording** is NOT needed

If the prompt doesn't appear, try `http://127.0.0.1:5000` instead of `localhost` — Chrome treats them differently for permissions.

## What your Colab work maps to

| Colab cells | VS Code file |
|-------------|--------------|
| Cell 1, 16, 53 (CMU dict) | `modules/speech_analysis.py::load_cmu_dict` |
| Cell 3 (phoneme_accuracy) | `modules/speech_analysis.py::phoneme_accuracy` |
| Cell 17 (apply_feedback) | `modules/speech_analysis.py::apply_feedback` |
| Cell 22, 49 (severity) | `modules/severity.py::estimate_severity` |
| Cell 30–35 (Allosaurus) | `modules/speech_analysis.py::predict_phonemes` |
| Cell 38 (preprocess) | `modules/speech_analysis.py::preprocess_audio` |
| Cell 42–43 (IPA→ARPA) | `modules/speech_analysis.py::convert_ipa_to_arpabet` |
| Cell 45 (extract_features) | `modules/speech_analysis.py::extract_features` |
| Cell 6, 9 (PPO networks) | `modules/rl_planner.py::Actor, Critic` |
| Phase 3 (new) | `modules/rl_planner.py::TherapyPlanner` |

## What's new vs Colab

1. **MediaPipe facial analysis** (`modules/facial_analysis.py`) — not in Colab yet
2. **SQLite persistence** — Colab had ephemeral `results` lists
3. **RL planner trains across sessions** — online updates every attempt, weights saved to `models/`
4. **Flask web GUI** — obvious
5. **Modular structure** — easier to test and extend

## Copying trained PPO weights from Colab (optional)

If you had a trained actor network from Colab Cell 9, you can drop it in:

```python
# In your Colab notebook, add this cell to save:
torch.save(actor.state_dict(), "actor.pt")
torch.save(critic.state_dict(), "critic.pt")
# Then download actor.pt and critic.pt from Colab
```

Move them to `models/ppo_actor.pt` and `models/ppo_critic.pt`. The planner will auto-load them on startup.

**Caveat:** the state dimensions differ between the two implementations. Colab used `INPUT_SIZE=10` (phoneme vector), VS Code uses `state_dim=5` (severity/acc/trend/sessions/fatigue). They're different agents doing different jobs — Colab's PPO picked which phoneme to correct; VS Code's PPO picks difficulty level. So you'll need to retrain anyway; the copy is optional.

## VS Code setup tips

Install these extensions:
- Python (Microsoft)
- Pylance
- Python Debugger
- SQLite Viewer (to browse `instance/therapy.db`)

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Flask: app.py",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/app.py",
      "console": "integratedTerminal"
    }
  ]
}
```

Now you can set breakpoints and F5 to debug Flask properly.

## Next steps after it runs

1. Add a patient, start a session, record one word — make sure the full loop works
2. Check `instance/therapy.db` (with SQLite Viewer) — you should see rows in `attempts`
3. Stop the server, restart — the RL planner weights should persist
4. Add more words to `THERAPY_PROTOCOLS` word pools for variety
5. Tune severity thresholds in `modules/severity.py` once you have real patient data
