"""
diagnose.py
════════════
Run this BEFORE doing anything else.
It checks the microphone, VAD, and model independently
so we can see exactly where the 0.000 problem comes from.

Usage:
  python diagnose.py --device 9
"""

import argparse
import time
import numpy as np

import config


def run(device_idx: int):
    import sounddevice as sd
    import openwakeword
    from openwakeword.utils import AudioFeatures
    import onnxruntime as ort

    # ── 1. Check microphone is delivering audio ───────────────────
    print("\n" + "─" * 60)
    print("  CHECK 1 — Microphone audio level")
    print("  Speak loudly for 3 seconds.  RMS should go above 500.")
    print("─" * 60)

    levels = []
    def _cb(indata, frames, t, status):
        rms = int(np.sqrt(np.mean((indata[:,0]*32768)**2)))
        levels.append(rms)
        bar = "█" * min(40, rms // 100)
        print(f"\r  RMS={rms:5d}  [{bar:<40}]", end="", flush=True)

    with sd.InputStream(device=device_idx, samplerate=16000,
                        channels=1, dtype="float32",
                        blocksize=1280, callback=_cb):
        time.sleep(3)
    print(f"\n  Peak RMS = {max(levels) if levels else 0}")

    if max(levels) < 100:
        print("\n  ✗  Mic is silent — no audio received.")
        print("     Check Linux audio permissions:")
        print("     sudo usermod -aG audio $USER  then re-login")
        return
    elif max(levels) < 300:
        print("  ⚠  Audio level low — try speaking closer or louder")
    else:
        print("  ✓  Mic is working")

    # ── 2. Silero VAD check ───────────────────────────────────────
    print("\n" + "─" * 60)
    print("  CHECK 2 — Silero VAD  (openWakeWord's voice activity detector)")
    print("  Speak 'bumblebee' — VAD score should go above 0.5")
    print("─" * 60)

    openwakeword.utils.download_models()
    from openwakeword.model import Model

    # Load model with VAD disabled so we see raw classifier
    onnx_path = config.MODEL_DIR / f"{config.MODEL_NAME}.onnx"
    oww = Model(wakeword_models=[str(onnx_path)],
                inference_framework="onnx",
                vad_threshold=0.0)

    vad_scores = []
    def _vad_cb(indata, frames, t, status):
        chunk = (indata[:,0] * 32768).astype(np.int16)
        # Access VAD score from model internals
        oww.predict(chunk)
        vad = float(oww.vad_scores.get("silero_vad", [0])[-1]) if hasattr(oww, "vad_scores") else 0.0
        vad_scores.append(vad)
        bar_v = "█" * int(vad * 30)
        print(f"\r  VAD={vad:.3f}  [{bar_v:<30}]", end="", flush=True)

    with sd.InputStream(device=device_idx, samplerate=16000,
                        channels=1, dtype="float32",
                        blocksize=1280, callback=_vad_cb):
        time.sleep(4)

    max_vad = max(vad_scores) if vad_scores else 0.0
    print(f"\n  Peak VAD = {max_vad:.3f}")
    if max_vad < 0.3:
        print("  ⚠  VAD never detected voice — it will block the classifier")
        print("     This is why score stays 0.000 in step5.")
        print("     Fix: set vad_threshold=0.0 in step5_test.py")
    else:
        print("  ✓  VAD is detecting voice")

    # ── 3. Raw classifier score  (VAD completely bypassed) ────────
    print("\n" + "─" * 60)
    print("  CHECK 3 — Raw classifier score  (NO VAD gate)")
    print("  Say 'bumblebee' several times.  Watch if score changes.")
    print("─" * 60)

    oww2 = Model(wakeword_models=[str(onnx_path)],
                 inference_framework="onnx",
                 vad_threshold=0.0)

    raw_scores = []
    def _raw_cb(indata, frames, t, status):
        chunk = (indata[:,0] * 32768).astype(np.int16)
        pred  = oww2.predict(chunk)
        score = float(list(pred.values())[0])
        raw_scores.append(score)
        bar_s = "█" * int(score * 40)
        print(f"\r  SCORE={score:.3f}  [{bar_s:<40}]", end="", flush=True)

    with sd.InputStream(device=device_idx, samplerate=16000,
                        channels=1, dtype="float32",
                        blocksize=1280, callback=_raw_cb):
        time.sleep(6)

    max_score = max(raw_scores) if raw_scores else 0.0
    min_score = min(raw_scores) if raw_scores else 0.0
    print(f"\n  Score range: {min_score:.3f} – {max_score:.3f}")

    # ── 4. Summary ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  DIAGNOSIS SUMMARY")
    print("═" * 60)

    if max(levels) < 100:
        print("  ✗ PROBLEM: Microphone not delivering audio")
        print("    FIX: sudo usermod -aG audio $USER  then re-login")
    elif max_vad < 0.3:
        print("  ✗ PROBLEM: VAD not detecting your voice")
        print("    FIX: In step5_test.py, change vad_threshold=0.5 to vad_threshold=0.0")
        print("         (line:  vad_threshold=0.5,  inside load_oww_model())")
    elif max_score < 0.3:
        print("  ✗ PROBLEM: Model scores real speech as 0")
        print("    ROOT CAUSE: Training data mismatch")
        print("    TTS embeddings ≠ real speech embeddings in the backbone space")
        print("    FIX: Retrain with TTS-only negatives (see instructions below)")
    elif max_score > 0.8:
        print("  ✓ Model is working! Adjust threshold in step5:")
        print(f"    python step5_test.py --device {device_idx} --threshold {max_score*0.8:.2f}")
    else:
        print(f"  ⚠ Model reacts but weakly (max={max_score:.3f})")
        print(f"    Try: python step5_test.py --device {device_idx} --threshold {max_score*0.7:.2f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True,
                        help="Microphone device index (use step5_test.py --list-devices)")
    args = parser.parse_args()
    run(args.device)