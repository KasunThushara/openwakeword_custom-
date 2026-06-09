"""
step5_test.py
══════════════
Tests your trained bumblebee.onnx model live with a microphone.

Modes
─────
  python step5_test.py                    →  live mic (default device)
  python step5_test.py --device 2         →  live mic (device index 2 = ReSpeaker)
  python step5_test.py --list-devices     →  print all audio devices + exit
  python step5_test.py --file clip.wav    →  score a single WAV file

Threshold
─────────
  python step5_test.py --device 2 --threshold 0.5
"""

import argparse
import time

import numpy as np
import soundfile as sf

import config


# ── List audio devices ────────────────────────────────────────────
def list_devices() -> None:
    import sounddevice as sd
    devices = sd.query_devices()
    print("\n  Audio input devices (use the index number with --device):\n")
    print(f"  {'IDX':>4}  {'NAME':<45}  {'IN CH':>6}  {'DEFAULT'}")
    print("  " + "─" * 70)
    default_in = sd.default.device[0]
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " ← DEFAULT" if i == default_in else ""
            respeaker = " 🎤 ReSpeaker" if "respeaker" in d["name"].lower() or "array" in d["name"].lower() else ""
            print(f"  {i:>4}  {d['name']:<45}  {d['max_input_channels']:>6}  {marker}{respeaker}")
    print()


# ── Load model ────────────────────────────────────────────────────
def load_model():
    onnx_path = config.MODEL_DIR / f"{config.MODEL_NAME}.onnx"
    if not onnx_path.exists():
        raise SystemExit(
            f"✗ Model not found: {onnx_path}\n"
            "  Run step4_train.py first."
        )
    try:
        import onnxruntime as ort
        from openwakeword.utils import AudioFeatures
    except ImportError:
        raise SystemExit("✗ pip install onnxruntime openwakeword")

    import openwakeword
    openwakeword.utils.download_models()
    
    print(f"  Loading embedding model (AudioFeatures)")
    F = AudioFeatures(device='cpu', ncpu=4)
    print(f"  ✓ AudioFeatures loaded")
    
    print(f"  Loading classifier: {onnx_path.name}")
    sess = ort.InferenceSession(str(onnx_path))
    print("  ✓ Classifier loaded")
    
    return F, sess


# ── File scoring mode ─────────────────────────────────────────────
def score_file(wav_path: str, threshold: float) -> None:
    F, sess = load_model()
    print(f"\n  Scoring: {wav_path}")
    audio, sr = sf.read(wav_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != config.SAMPLE_RATE:
        import librosa
        audio = (librosa.resample(audio.astype(np.float32) / 32768.0,
                                  orig_sr=sr, target_sr=config.SAMPLE_RATE) * 32768).astype(np.int16)
    
    # Extract embeddings
    emb = F.embed_clips(np.array([audio]), batch_size=1)
    
    # Extract window from center (wake word should be in the middle)
    n_frames = emb.shape[1]
    W = config.WINDOW_FRAMES
    if n_frames >= W:
        start = max(0, (n_frames - W) // 2)
        window = emb[0, start : start + W, :]
    else:
        pad = W - n_frames
        window = np.pad(emb[0], ((0, pad), (0, 0)), mode="constant")
    
    # Score
    test_input = window[np.newaxis, :, :].astype(np.float32)
    result = sess.run(None, {"input": test_input})
    max_score = float(result[0][0][0])
    
    print(f"\n  Peak score : {max_score:.4f}  (threshold={threshold})")
    print("  🐝 DETECTED" if max_score >= threshold else "  — not detected")


# ── Live mic mode ─────────────────────────────────────────────────
def live_test(threshold: float, device) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("✗ pip install sounddevice")

    F, sess = load_model()

    # ── Show which device we're using ─────────────────────────────
    if device is None:
        dev_info = sd.query_devices(kind="input")
    else:
        # Accept both int index and partial name string
        if isinstance(device, str) and not device.lstrip("-").isdigit():
            # partial name match
            all_devs = sd.query_devices()
            matches = [i for i, d in enumerate(all_devs)
                       if device.lower() in d["name"].lower()
                       and d["max_input_channels"] > 0]
            if not matches:
                raise SystemExit(f"✗ No device matching '{device}' found.\n"
                                 "  Run with --list-devices to see options.")
            device = matches[0]
        device = int(device)
        dev_info = sd.query_devices(device)

    print(f"\n  🎙  Device : [{device if device is not None else 'default'}] {dev_info['name']}")
    print(f"      Channels available : {dev_info['max_input_channels']}")

    # ReSpeaker has 4 channels – we only need 1 (channel 0 = mixed)
    n_ch = min(dev_info["max_input_channels"], 1)

    CHUNK   = 1280   # 80 ms at 16 kHz
    SILENCE = 0.5    # seconds between detections

    print(f"\n  🎤  Listening … say 'bumblebee'  (threshold={threshold})")
    print("  Ctrl+C to stop\n")

    last_detect = 0.0
    audio_buffer = np.zeros((0,), dtype=np.int16)
    # Use a fixed 1.28s audio chunk for scoring.
    # This matches the classifier's WINDOW_FRAMES size after padding.
    MIN_EMBED_SAMPLES = config.SAMPLE_RATE * 1280 // 1000  # 20480 samples
    MAX_BUFFER_SAMPLES = MIN_EMBED_SAMPLES * 2
    score_history = []
    SMOOTH_FRAMES = 3

    def callback(indata, frames, t, status):
        nonlocal last_detect, audio_buffer, score_history
        if status:
            # Print ALSA warnings only once so they don't spam the bar
            pass
        # Mix to mono: take channel 0 (already mono if n_ch=1)
        mono = indata[:, 0]
        chunk = (mono * 32768).astype(np.int16)

        audio_buffer = np.concatenate((audio_buffer, chunk))
        if len(audio_buffer) < MIN_EMBED_SAMPLES:
            return
        if len(audio_buffer) > MAX_BUFFER_SAMPLES:
            audio_buffer = audio_buffer[-MAX_BUFFER_SAMPLES:]

        samples = audio_buffer[-MIN_EMBED_SAMPLES:]

        # Extract embeddings for the latest fixed chunk of audio
        emb = F.embed_clips(np.array([samples]), batch_size=1)

        n_frames = emb.shape[1]
        W = config.WINDOW_FRAMES
        if n_frames >= W:
            # For longer chunks, use the centre window to mimic training positives.
            start = max(0, (n_frames - W) // 2)
            window = emb[0, start:start + W, :]
        else:
            pad = W - n_frames
            window = np.pad(emb[0], ((0, pad), (0, 0)), mode="constant")

        test_input = window[np.newaxis, :, :].astype(np.float32)
        result = sess.run(None, {"input": test_input})
        score = float(result[0][0][0])

        score_history.append(score)
        if len(score_history) > SMOOTH_FRAMES:
            score_history.pop(0)
        avg_score = float(np.mean(score_history))

        bar_len = int(avg_score * 40)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        print(f"\r  [{bar}]  {avg_score:.3f}", end="", flush=True)

        now = time.time()
        if avg_score >= threshold and score >= threshold and (now - last_detect) > SILENCE:
            last_detect = now
            print(f"\n  🐝  BUMBLEBEE detected!  score={avg_score:.3f}")

    try:
        with sd.InputStream(
            device=device,
            samplerate=config.SAMPLE_RATE,
            channels=1,           # always request 1 channel (ALSA mixes for us)
            dtype="float32",
            blocksize=CHUNK,
            callback=callback,
        ):
            while True:
                sd.sleep(100)
    except sd.PortAudioError as e:
        print(f"\n\n  ✗ PortAudio error: {e}")
        print("  Try:  python step5_test.py --list-devices")
        print("  Then: python step5_test.py --device <index>")
    except KeyboardInterrupt:
        print("\n\n  Stopped.")


# ── Arg parser ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test bumblebee wake word",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--device", default=None,
        help=(
            "Audio input device index or partial name.\n"
            "Examples:\n"
            "  --device 2          (use index from --list-devices)\n"
            "  --device respeaker  (partial name match)\n"
            "  (omit to use system default)"
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection threshold 0–1  (default: 0.5)",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Score a WAV file instead of live mic",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print all audio input devices and exit",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 5 – Live microphone test")
    print("=" * 60)

    if args.list_devices:
        list_devices()
    elif args.file:
        score_file(args.file, args.threshold)
    else:
        live_test(args.threshold, args.device)