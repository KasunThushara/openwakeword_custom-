"""
step5_test.py  (FIXED)
══════════════════════
Key fix: use openWakeWord's Model.predict() with 80 ms chunks.
This maintains an internal rolling buffer of the last 16 embedding frames —
exactly matching how the model was trained.

The previous version embedded long chunks manually, which produced a
different distribution from training → model scored 1.000 on everything.

Usage
─────
  python step5_test.py --list-devices
  python step5_test.py --device 9
  python step5_test.py --device 9 --threshold 0.6
  python step5_test.py --file data/positive/bumblebee_00001.wav
"""

import argparse
import shutil
import time

import numpy as np
import soundfile as sf

import config


# ── List audio devices ────────────────────────────────────────────
def list_devices() -> None:
    import sounddevice as sd
    devices = sd.query_devices()
    print("\n  Audio input devices:\n")
    print(f"  {'IDX':>4}  {'NAME':<48}  {'IN CH':>6}")
    print("  " + "─" * 65)
    default_in = sd.default.device[0]
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            tag = ""
            if i == default_in:
                tag += " ← DEFAULT"
            if "respeaker" in d["name"].lower() or "array" in d["name"].lower():
                tag += " 🎤 ReSpeaker"
            print(f"  {i:>4}  {d['name']:<48}  {d['max_input_channels']:>6}  {tag}")
    print()


# ── Resolve device argument → int index ──────────────────────────
def resolve_device(device) -> int | None:
    if device is None:
        return None
    if isinstance(device, str) and not device.lstrip("-").isdigit():
        import sounddevice as sd
        matches = [i for i, d in enumerate(sd.query_devices())
                   if device.lower() in d["name"].lower()
                   and d["max_input_channels"] > 0]
        if not matches:
            raise SystemExit(f"✗ No input device matching '{device}'.\n"
                             "  Run --list-devices to see options.")
        return matches[0]
    return int(device)


# ── Load model ────────────────────────────────────────────────────
def load_oww_model():
    """
    Load via openWakeWord's Model class.
    Model.predict(chunk) maintains a rolling 16-frame embedding buffer
    internally — this is the correct way to do streaming inference.
    """
    onnx_path = config.MODEL_DIR / f"{config.MODEL_NAME}.onnx"
    if not onnx_path.exists():
        default_path = config.DEFAULT_MODEL_DIR / f"{config.MODEL_NAME}.onnx"
        if default_path.exists():
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(default_path, onnx_path)
            print(f"  Copied default model to {onnx_path}")
        else:
            raise SystemExit(f"✗ Model not found: {onnx_path}\n"
                             "  Run step4_train.py first.")

    import openwakeword
    from openwakeword.model import Model

    openwakeword.utils.download_models()

    print(f"  Loading: {onnx_path.name}")
    model = Model(
        wakeword_models=[str(onnx_path)],
        inference_framework="onnx",
        vad_threshold=0.0,    # disable VAD gate so model scores every chunk
    )
    print("  ✓ Model loaded")
    print("  VAD threshold : 0.0  (VAD disabled)")
    return model


# ── File scoring mode ─────────────────────────────────────────────
def score_file(wav_path: str, threshold: float) -> None:
    model = load_oww_model()
    print(f"\n  Scoring: {wav_path}")

    audio, sr = sf.read(wav_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != config.SAMPLE_RATE:
        import librosa
        audio = (librosa.resample(audio.astype(np.float32) / 32768.0,
                                  orig_sr=sr, target_sr=config.SAMPLE_RATE
                                  ) * 32768).astype(np.int16)

    CHUNK = 1280
    max_score = 0.0
    for i in range(0, len(audio) - CHUNK, CHUNK):
        pred  = model.predict(audio[i:i+CHUNK])
        score = float(list(pred.values())[0])
        max_score = max(max_score, score)

    print(f"\n  Peak score : {max_score:.4f}  (threshold={threshold})")
    print("  🐝 BUMBLEBEE DETECTED" if max_score >= threshold else "  — not detected")


# ── Live mic mode ─────────────────────────────────────────────────
def live_test(threshold: float, device) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("✗ pip install sounddevice")

    device = resolve_device(device)
    model  = load_oww_model()

    if device is None:
        dev_info = sd.query_devices(kind="input")
    else:
        dev_info = sd.query_devices(device)

    print(f"\n  🎙  Device  : [{device}] {dev_info['name']}")
    print(f"  Channels   : {dev_info['max_input_channels']}")
    print(f"  Threshold  : {threshold}")
    print(f"\n  🎤  Listening … say 'bumblebee'")
    print("  Ctrl+C to stop\n")

    # ── How Model.predict() works (for understanding) ──────────────
    # Each call receives 1280 int16 samples (80 ms at 16 kHz).
    # Internally it:
    #   1. Runs the melspectrogram model on those 80 ms
    #   2. Runs the Google embedding model → 1 new embedding frame (96-dim)
    #   3. Appends to a rolling buffer of the last 16 frames
    #   4. Runs our classifier on the 16-frame buffer
    #   5. Returns score
    # This exactly matches the training distribution.

    CHUNK   = 1280       # 80 ms — must match OWW's expected chunk size
    SILENCE = 1.0        # min seconds between detections (debounce)
    last_detect = 0.0

    def callback(indata, frames, t, status):
        nonlocal last_detect

        # Convert float32 sounddevice audio → int16 (OWW expects int16)
        chunk = (indata[:, 0] * 32768).astype(np.int16)

        # predict() feeds into rolling buffer and scores
        pred  = model.predict(chunk)
        score = float(list(pred.values())[0])

        # Score bar
        bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
        print(f"\r  [{bar}]  {score:.3f}", end="", flush=True)

        now = time.time()
        if score >= threshold and (now - last_detect) > SILENCE:
            last_detect = now
            model.reset()   # clear rolling buffer to prevent echo detections
            print(f"\n  🐝  BUMBLEBEE!  score={score:.3f}")

    try:
        with sd.InputStream(
            device=device,
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK,
            callback=callback,
        ):
            while True:
                sd.sleep(100)
    except sd.PortAudioError as e:
        print(f"\n\n  ✗ PortAudio error: {e}")
        print("  Try: python step5_test.py --list-devices")
    except KeyboardInterrupt:
        print("\n\n  Stopped.")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test bumblebee wake word",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--device", default=None,
        help="Device index or partial name  (e.g. --device 9  or  --device respeaker)")
    parser.add_argument("--threshold", type=float, default=0.5,
        help="Detection threshold 0–1  (default 0.5)")
    parser.add_argument("--file", type=str, default=None,
        help="Score a WAV file instead of live mic")
    parser.add_argument("--list-devices", action="store_true",
        help="Print all audio input devices and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 5 – Live microphone test  (FIXED)")
    print("=" * 60)

    if args.list_devices:
        list_devices()
    elif args.file:
        score_file(args.file, args.threshold)
    else:
        live_test(args.threshold, args.device)