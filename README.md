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
├── requirements.txt
├── step1_generate_clips.py    # Generate 5,000 TTS positive clips
├── step2_get_negatives.py     # Generate 3,000 TTS negative clips
├── step3_extract_features.py  # Compute openWakeWord embeddings
├── step4_train.py             # Train classifier and export ONNX
├── step5_test.py              # Live microphone test
├── diagnose.py                # Debug microphone/VAD/model issues
├── data/
│   ├── positive/
│   └── negative/
├── features/
├── model/
│   └── bumblebee.onnx
└── voices/
```

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

Run the scripts in order.

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
