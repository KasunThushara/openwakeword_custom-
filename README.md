# Bumblebee Wake Word — openWakeWord Training Pipeline

Custom wake word **"bumblebee"** trained locally using openWakeWord.

No Colab, no cloud, no Jupyter — pure Python scripts in PyCharm.

---

## System

| Item       | Value                               |
| ---------- | ----------------------------------- |
| OS         | Ubuntu (x86_64)                     |
| Python     | 3.10                                |
| IDE        | PyCharm                             |
| Hardware   | CPU only, 32 GB RAM                 |
| Microphone | Seeed ReSpeaker XVF3800 4-Mic Array |

---

## Project Structure

```text
bumblebee_wakeword/
├── config.py                  # All settings in one place
├── app.py                     # Flask web UI for training
├── bumblebee_server.py        # WebSocket wake-word detection server
├── bumblebee_face.html        # Animated bot face (HTML + SVG)
├── requirements.txt
├── step1_generate_clips.py    # Generate 5,000 TTS positive clips
├── step2_get_negatives.py     # Generate 3,000 TTS negative clips
├── step3_extract_features.py  # Compute openWakeWord embeddings
├── step4_train.py             # Train classifier and export ONNX
├── step5_test.py              # Live microphone test (CLI mode)
├── diagnose.py                # Debug microphone/VAD/model issues
├── data/
│   ├── positive/              # Wake-word WAV clips
│   └── negative/              # Non-wake-word WAV clips
├── features/
│   ├── positive_train.npy
│   ├── positive_val.npy
│   ├── negative_train.npy
│   └── negative_val.npy
├── model/
│   └── bumblebee.onnx         # Trained ONNX classifier
└── voices/
    ├── en_US-amy-medium.onnx
    ├── en_US-amy-medium.onnx.json
    ├── en_US-lessac-medium.onnx
    └── en_US-lessac-medium.onnx.json
```

---

# Quick Start (Web UI)

The fastest way to train a custom wake word — no command-line needed.

```bash
pip install flask
python app.py
```

Open browser: **http://localhost:5000**

---

# Installation

Install packages in this order. Installing openWakeWord first may downgrade PyTorch.

## 1. Install PyTorch (CPU Build)

```bash
pip install torch==2.2.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu
```

## 2. Install openWakeWord

```bash
pip install openwakeword
```

## 3. Pin NumPy Below 2.0

Required for Ubuntu + Python 3.10 when using Torch 2.2.2.

```bash
pip install "numpy<2.0"
```

## 4. Install Remaining Dependencies

```bash
pip install piper-tts soundfile librosa audiomentations \
            datasets tqdm scikit-learn pyyaml requests \
            onnx onnxruntime sounddevice
```

---

## Verify Installation

```bash
python -c "
import numpy, torch, openwakeword, piper
print('numpy :', numpy.__version__)
print('torch :', torch.__version__)
arr = torch.from_numpy(numpy.zeros(10))
print('bridge: OK')
"
```

Expected:

```text
numpy : 1.26.x
torch : 2.2.2+cpu
bridge: OK
```

---

# Running the Pipeline

Run the scripts in order. You can use **either**:

- **Command-line** (scripts)  — For automation and power users
- **Web UI** (Flask) — For interactive training with visual feedback

See [Web UI — Flask Trainer](#web-ui--flask-trainer) for the browser-based approach.

---

## Step 1 — Generate Positive Clips

Downloads Piper voices and synthesizes approximately 5,000 wake-word samples.

```bash
python step1_generate_clips.py
```

Output:

```text
data/positive/*.wav
```

---

## Step 2 — Generate Negative Clips

Creates approximately 3,000 non-wake-word samples using the same TTS voices.

```bash
python step2_get_negatives.py
```

### Why Use TTS Negatives?

If real speech is used for negatives while TTS is used for positives, the model may learn:

```text
TTS speech = wake word
Real speech = not wake word
```

Using TTS for both classes forces the model to learn phonetics rather than recording style.

---

## Step 3 — Extract Features

Computes frozen speech embeddings for every WAV file.

```bash
python step3_extract_features.py
```

Runtime:

```text
~45 minutes on CPU
```

Outputs:

```text
features/*.npy
```

If interrupted:

```bash
rm -rf features
python step3_extract_features.py
```

---

## Step 4 — Train and Export Model

Train the classifier and export ONNX.

```bash
python step4_train.py
```

Expected training progress:

```text
Epoch   1/60  tr_loss=0.58  va_loss=0.54  tr_acc=72%  va_acc=74%
Epoch  10/60  tr_loss=0.31  va_loss=0.28  tr_acc=87%  va_acc=88%
Epoch  30/60  tr_loss=0.14  va_loss=0.17  tr_acc=94%  va_acc=92%
```

Output:

```text
model/bumblebee.onnx
```

---

## Step 5 — Live Microphone Test

List devices:

```bash
python step5_test.py --list-devices
```

Run using ReSpeaker device index:

```bash
python step5_test.py --device 9
```

---

# Known Issues and Fixes

## PyTorch Downgraded After Installing openWakeWord

```bash
pip install torch==2.2.2 --force-reinstall
```

---

## NumPy 2.x Error

Error:

```text
_ARRAY_API not found
```

Fix:

```bash
pip install "numpy<2.0"
```

---

## Score Stuck at 0.000

Cause:

```text
Silero VAD blocking all speech
```

Edit:

```python
vad_threshold=0.5
```

Change to:

```python
vad_threshold=0.0
```

This disables VAD gating.

---

## Score Stuck at 1.000

Cause:

```text
Model overfit to recording style
```

Typical reason:

```text
Real speech negatives + TTS positives
```

Fix:

1. Regenerate TTS negatives.
2. Delete features.
3. Re-run Steps 3 and 4.

---

## Diagnose Problems

```bash
python diagnose.py --device 9
```

Checks:

* Microphone level
* VAD output
* Raw classifier score

---

# Tuning

All configuration values live in:

```text
config.py
```

| Setting          | Default    | Recommendation                                  |
| ---------------- | ---------- | ----------------------------------------------- |
| N_POSITIVE_CLIPS | 5000       | Increase to 8000 if accuracy is low             |
| EPOCHS           | 60         | Increase if validation loss is still decreasing |
| HIDDEN_SIZE      | 128        | Try 64 or 256                                   |
| PHRASES          | 5 variants | Add realistic pronunciations                    |

---

## What To Re-Run After Changes

| Changed Setting  | Re-run                   |
| ---------------- | ------------------------ |
| PHRASES          | Step 1 → Step 3 → Step 4 |
| N_POSITIVE_CLIPS | Step 1 → Step 3 → Step 4 |
| N_NEGATIVE_CLIPS | Step 2 → Step 3 → Step 4 |
| EPOCHS           | Step 4 only              |
| HIDDEN_SIZE      | Step 4 only              |
| LEARNING_RATE    | Step 4 only              |

---

# Using the Trained Model

```python
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

model = Model(
    wakeword_models=["model/bumblebee.onnx"],
    inference_framework="onnx",
    vad_threshold=0.0,
)

def on_wake_word():
    print("🐝 Bumblebee detected!")
    model.reset()

def callback(indata, frames, t, status):
    chunk = (indata[:, 0] * 32768).astype(np.int16)

    score = float(
        list(model.predict(chunk).values())[0]
    )

    if score >= 0.5:
        on_wake_word()

with sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype="float32",
    blocksize=1280,
    callback=callback,
):
    input("Listening... press Enter to stop\n")
```

---

# Web UI — Flask Trainer

A modern web interface for training and testing your custom wake word without command-line scripts.

## Installation

```bash
pip install flask
```

## Running the UI

```bash
python app.py
```

Open your browser:

```text
http://localhost:5000
```

---

## Features

### 🎛️ Configuration Panel

- **Wake Word**: Set the keyword (e.g., "bumblebee")
- **Phrase Variants**: Add pronunciation variations for TTS synthesis
- **Dataset Sizes**: Control positive and negative clip counts
- **Model Hyperparameters**:
  - Hidden layer size (default: 128)
  - Training epochs (default: 60)
  - Learning rate (default: 0.001)

All changes are saved to `config.py` in real-time.

---

### ⚙️ Pipeline Control

Run each training step with a single click:

1. **Generate Positive Clips** (~5–15 min)
   - Synthesizes wake-word samples using Piper TTS
   - Two diverse voices (male + female)

2. **Generate Negative Clips** (~3–10 min)
   - Creates non-wake-word samples using same TTS voices
   - Prevents recording-style overfitting

3. **Extract Features** (~45 min)
   - Computes openWakeWord embeddings for all audio
   - CPU-optimized batch processing

4. **Train & Export Model** (~5–15 min)
   - Trains classifier on embeddings
   - Exports production-ready ONNX model

Each step shows:
- Real-time log output
- Elapsed time
- Status (pending / running / done / error)
- One-click stop button

---

### 🔧 Maintenance & Cleanup

Delete intermediate files without touching source:

- **Clear Data**: Removes all WAV clips
- **Clear Features**: Removes pre-computed embeddings (forces Step 3 re-run)
- **Clear Model**: Removes ONNX export (forces Step 4 re-run)

---

### 🎙️ Live Wake-Word Server

Start the WebSocket server from the UI:

- **Device Selection**: Choose microphone from dropdown
- **Threshold Slider**: Adjust detection sensitivity (0.0–1.0)
- **Start/Stop Server**: Control listening without leaving the browser
- **Face Animation**: Watch the animated bot react to wake-word detections

Connects to `bumblebee_server.py` and drives the animated face in [bumblebee_face.html](#face-animation).

---

### 📊 Status Dashboard

Real-time metrics:

- ✅ Model exists (green dot when ready)
- 📁 Positive clips count
- 📁 Negative clips count
- 📊 Features computed (all .npy files present)
- ⏱️ Step completion times

---

## Example Workflow

1. **Configure** → Set wake word to "hello" + 3 phrases
2. **Generate** → Click "Run Step 1" (clips) → "Run Step 2" (negatives)
3. **Extract** → Click "Run Step 3" (wait ~45 min)
4. **Train** → Click "Run Step 4" (classifier trained)
5. **Monitor** → Start server and speak "hello" into mic
6. **Watch** → Face animates when wake word is detected

---

## API Endpoints

If you want to automate or integrate with other tools:

```text
POST   /api/config          — Read/write config.py settings
POST   /api/run/<step>      — Start training step (1–4)
GET    /api/run/<step>/stream — Stream step output (Server-Sent Events)
POST   /api/run/<step>/stop — Stop running step
GET    /api/status          — Check dataset & model status
DELETE /api/clean/<target>  — Delete data/features/model
GET    /api/devices         — List audio input devices
POST   /api/server/start    — Start wake-word detection server
POST   /api/server/stop     — Stop detection server
GET    /api/server/status   — Server running status
GET    /face                — Get animated bot face (with keyword substitution)
```

---

## Face Animation

The web UI displays an animated Autobot-style face that reacts to wake-word detections.

When a detection occurs:
- Eyes glow **blue** 🔵
- Horns glow **yellow** 🟡
- Scan lines animate
- Speaker grille bars animate
- Status text flashes "BUMBLEBEE!"

Customize the face by editing:

```text
bumblebee_face.html
```

The face keywords are auto-substituted based on your `WAKE_WORD` in config.py.

---

## Troubleshooting

### UI Won't Load

```bash
pip install flask
python app.py
```

If port 5000 is in use:

```bash
lsof -i :5000
kill -9 <PID>
```

---

### Steps Keep Failing

Check the step's **log output** in the UI for detailed error messages.

Common issues:

- Missing dependencies → `pip install -r requirements.txt`
- Disk space → `df -h`
- PyTorch issues → `pip install torch==2.2.2 --force-reinstall`

---

### Server Won't Start

1. Model must exist (run Step 4 first)
2. Microphone must be detected (check dropdown)
3. Check firewall (port 8767 for WebSocket)

Run diagnosis from terminal:

```bash
python diagnose.py --device 9
```

---

# ESP32-S3 Deployment

The exported ONNX model is too large for ESP32-S3 devices.

For ESPHome deployment, use microWakeWord:

* Quantized INT8 model
* Approximately 35 KB
* Optimized for ESP32-S3

The generated WAV datasets can be reused:

```text
data/positive/
data/negative/
```

No need to regenerate audio.

Only retrain using the microWakeWord pipeline.

---

# References

openWakeWord:
https://github.com/dscripka/openWakeWord

Piper TTS:
https://huggingface.co/rhasspy/piper-voices

microWakeWord:
https://github.com/OHF-Voice/micro-wake-word

ESPHome micro_wake_word:
https://esphome.io/components/micro_wake_word/
