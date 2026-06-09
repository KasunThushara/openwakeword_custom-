"""
step5_test.py
══════════════
Tests your trained bumblebee.onnx model live with a microphone.

Two modes
─────────
  python step5_test.py            →  live mic test (default)
  python step5_test.py --file x   →  score a single WAV file

Controls
────────
  Ctrl+C   stop listening
  --threshold 0.5   adjust sensitivity (lower = more sensitive)

Requirements
────────────
  pip install sounddevice   (easier than pyaudio on Windows)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import config


def load_model():
    """Load the trained ONNX model via openWakeWord."""
    onnx_path = config.MODEL_DIR / f"{config.MODEL_NAME}.onnx"
    if not onnx_path.exists():
        raise SystemExit(
            f"✗ Model not found: {onnx_path}\n"
            "  Run step4_train.py first."
        )

    try:
        from openwakeword.model import Model
    except ImportError:
        raise SystemExit("✗ Install: pip install openwakeword")

    # Ensure backbone models are present
    import openwakeword
    openwakeword.utils.download_models()

    print(f"  Loading model: {onnx_path}")
    model = Model(
        wakeword_models=[str(onnx_path)],
        inference_framework="onnx",
        vad_threshold=0.0,    # we handle threshold ourselves for clarity
    )
    print("  ✓ Model loaded")
    return model


# ── File mode: score a single WAV ─────────────────────────────────
def score_file(wav_path: str, threshold: float) -> None:
    model = load_model()

    print(f"\n  Scoring: {wav_path}")
    audio, sr = sf.read(wav_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != config.SAMPLE_RATE:
        import librosa
        audio_f = audio.astype(np.float32) / 32768.0
        audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        audio   = (audio_f * 32768).astype(np.int16)

    CHUNK = 1280    # 80 ms
    max_score = 0.0
    for i in range(0, len(audio) - CHUNK, CHUNK):
        chunk = audio[i : i + CHUNK]
        pred  = model.predict(chunk)
        score = float(list(pred.values())[0])
        max_score = max(max_score, score)

    print(f"\n  Peak score: {max_score:.4f}  (threshold = {threshold})")
    if max_score >= threshold:
        print("  🐝 BUMBLEBEE DETECTED")
    else:
        print("  — Not detected")


# ── Live mode ─────────────────────────────────────────────────────
def live_test(threshold: float) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit(
            "✗ sounddevice not found.\n"
            "  Run:  pip install sounddevice\n"
            "  (On Windows you may also need:  pip install sounddevice --pre)\n"
        )

    model = load_model()

    CHUNK   = 1280          # 80 ms at 16 kHz = 1280 samples
    SILENCE = 0.5           # minimum seconds between detections

    print(f"\n  🎤  Listening … say 'bumblebee'  (threshold={threshold})")
    print("  Ctrl+C to stop\n")

    last_detect = 0.0
    audio_buffer = np.zeros(CHUNK, dtype=np.int16)

    def callback(indata, frames, t, status):
        nonlocal last_detect
        # indata is float32 from sounddevice; convert to int16
        chunk = (indata[:, 0] * 32768).astype(np.int16)
        pred  = model.predict(chunk)
        score = float(list(pred.values())[0])

        # Visual bar
        bar_len = int(score * 40)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        print(f"\r  [{bar}]  {score:.3f}", end="", flush=True)

        now = time.time()
        if score >= threshold and (now - last_detect) > SILENCE:
            last_detect = now
            print(f"\n  🐝  BUMBLEBEE! (score={score:.3f})")

    try:
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK,
            callback=callback,
        ):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\n\n  Stopped.")


# ── Threshold helper ──────────────────────────────────────────────
def print_threshold_guide() -> None:
    print("""
  Threshold tuning guide
  ──────────────────────
  0.3 – 0.4   High sensitivity. More detections, more false positives.
               Good for: noisy rooms, far-field use
  0.5         Balanced default. Start here.
  0.6 – 0.8   Conservative.  Fewer false positives, may miss at distance.
               Good for: quiet environments, close mic

  If you get false positives: raise threshold (e.g. --threshold 0.6)
  If it misses your voice:    lower threshold (e.g. --threshold 0.4)
    """)


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 5 – Live microphone test")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Test bumblebee wake word")
    parser.add_argument("--file",      type=str,   default=None,
                        help="Path to a WAV file to score (instead of mic)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold 0–1  (default: 0.5)")
    args = parser.parse_args()

    print_threshold_guide()

    if args.file:
        score_file(args.file, args.threshold)
    else:
        live_test(args.threshold)
