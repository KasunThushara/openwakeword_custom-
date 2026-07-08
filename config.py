"""
config.py  –  Single place to change anything about the project.
All other scripts import from here.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
DATA_DIR      = PROJECT_ROOT / "data"
POSITIVE_DIR  = DATA_DIR / "positive"
NEGATIVE_DIR  = DATA_DIR / "negative"
FEATURES_DIR  = PROJECT_ROOT / "features"
MODEL_DIR     = PROJECT_ROOT / "model"
VOICES_DIR    = PROJECT_ROOT / "voices"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "default_model"

# auto-create all dirs so scripts can be run in any order
for _d in [POSITIVE_DIR, NEGATIVE_DIR, FEATURES_DIR, MODEL_DIR, VOICES_DIR, DEFAULT_MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Wake word ──────────────────────────────────────────────────────
WAKE_WORD = "wave your antenna"

# Phrase variants fed to TTS.
# The model learns ALL of them – use phrases you'll actually speak.
PHRASES = [
    "wave your antenna",
]

# ── Audio ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000       # Hz  – openWakeWord always uses 16 kHz

# ── Data volumes ───────────────────────────────────────────────────
N_POSITIVE_CLIPS = 5000   # synthetic TTS clips  (aim for 5k minimum)
N_NEGATIVE_CLIPS = 3000   # from LibriSpeech streaming

# ── Embedding / model shape ────────────────────────────────────────
# openWakeWord produces 96-dim embeddings every 80 ms.
# The classifier sees a rolling window of WINDOW_FRAMES frames.
WINDOW_FRAMES  = 16        # 16 × 80 ms = 1.28 seconds
EMBEDDING_DIM  = 96

# ── Training hyper-parameters ──────────────────────────────────────
HIDDEN_SIZE    = 128
BATCH_SIZE     = 256
EPOCHS         = 60
LEARNING_RATE  = 0.001
VAL_SPLIT      = 0.15      # 15 % held out for validation

# ── Output ─────────────────────────────────────────────────────────
MODEL_NAME     = "wave_your_antenna"   # output file: model/bumblebee.onnx

# ── Music / sound effects ──────────────────────────────────────────
MUSIC_DIR       = PROJECT_ROOT / "music"
WAKE_SOUND      = MUSIC_DIR / "robot.wav"
COMMAND_TIMEOUT = 6           # seconds to wait for a command after wake

# ── Piper voice models to download ────────────────────────────────
# Using two voices (male + female) for speaker diversity.
# Both are ~60 MB each – downloaded once by step1.
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICE_MODELS = [
    {
        "name": "en_US-lessac-medium",
        "onnx": f"{HF_BASE}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": f"{HF_BASE}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    {
        "name": "en_US-amy-medium",
        "onnx": f"{HF_BASE}/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "json": f"{HF_BASE}/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    },
]
