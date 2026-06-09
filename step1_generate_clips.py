"""
step1_generate_clips.py
═══════════════════════
Generates synthetic "bumblebee" speech clips using Piper TTS.

What it does
------------
1. Downloads two Piper voice models into voices/  (≈ 60 MB each, once only)
2. Synthesises N_POSITIVE_CLIPS WAV files at 16 kHz into data/positive/
3. Applies random pitch/speed augmentation so the model is robust

Expected run time on CPU:  ~8–15 minutes for 5 000 clips
"""

import io
import random
import urllib.request
import wave
from pathlib import Path

import audiomentations as A
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

import config


# ── Augmentation pipeline ──────────────────────────────────────────
# Applied to every generated clip.  Keeps the word intelligible but
# adds the variation a real model needs to be robust.
augment = A.Compose([
    A.PitchShift(min_semitones=-3, max_semitones=3, p=0.6),
    A.TimeStretch(min_rate=0.85, max_rate=1.20, p=0.6),
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.012, p=0.4),
    A.LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=6000, p=0.2),
])


# ── Helper: download a file with a progress bar ────────────────────
def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  ✓ already exists: {dest.name}")
        return
    print(f"  ↓ downloading {dest.name} …")
    dest.parent.mkdir(parents=True, exist_ok=True)

    class _Progress:
        def __init__(self):
            self.bar = None
        def __call__(self, block, block_size, total):
            if self.bar is None:
                self.bar = tqdm(total=total, unit="B", unit_scale=True, leave=False)
            self.bar.update(block_size)

    urllib.request.urlretrieve(url, dest, reporthook=_Progress())
    print(f"  ✓ saved to {dest}")


# ── Step 1-A : Download voices ─────────────────────────────────────
def download_voices() -> list[Path]:
    """Return list of downloaded .onnx voice paths."""
    onnx_paths = []
    print("\n[Step 1-A] Downloading Piper voice models …")
    for vm in config.VOICE_MODELS:
        onnx_path = config.VOICES_DIR / f"{vm['name']}.onnx"
        json_path = config.VOICES_DIR / f"{vm['name']}.onnx.json"
        download(vm["onnx"], onnx_path)
        download(vm["json"], json_path)
        onnx_paths.append(onnx_path)
    return onnx_paths


# ── Step 1-B : Synthesise clips ────────────────────────────────────
def synthesise_clips(voice_paths: list[Path]) -> None:
    """Use Piper to generate N_POSITIVE_CLIPS WAV files."""
    # Import here so that users see a clear error if piper-tts is missing
    try:
        from piper.voice import PiperVoice
    except ImportError:
        raise SystemExit(
            "\n✗ piper-tts not found.\n"
            "  Run:  pip install piper-tts\n"
            "  Then re-run this script.\n"
        )

    print(f"\n[Step 1-B] Loading {len(voice_paths)} Piper voice(s) …")
    voices = [PiperVoice.load(str(p)) for p in voice_paths]
    print(f"  ✓ voices loaded")

    print(f"\n[Step 1-B] Generating {config.N_POSITIVE_CLIPS} positive clips …")
    print(f"  Phrases: {config.PHRASES}")
    print(f"  Output:  {config.POSITIVE_DIR}\n")

    existing = list(config.POSITIVE_DIR.glob("*.wav"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"  ↷ {start_idx} clips already exist – resuming from {start_idx}")

    for i in tqdm(range(start_idx, config.N_POSITIVE_CLIPS), unit="clip"):
        voice  = random.choice(voices)
        phrase = random.choice(config.PHRASES)

        # ── Synthesise ──────────────────────────────────────────────
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(phrase, wf)

        # ── Decode WAV bytes → numpy float32 ────────────────────────
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            src_sr     = wf.getframerate()
            raw_bytes  = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:                       # keep mono only
            audio = audio.reshape(-1, n_channels)[:, 0]

        # ── Resample to 16 kHz if the voice outputs at different rate ─
        if src_sr != config.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=src_sr, target_sr=config.SAMPLE_RATE)

        # ── Augment ─────────────────────────────────────────────────
        audio = augment(samples=audio, sample_rate=config.SAMPLE_RATE)

        # ── Save ────────────────────────────────────────────────────
        out = config.POSITIVE_DIR / f"bumblebee_{i:05d}.wav"
        sf.write(str(out), audio, config.SAMPLE_RATE, subtype="PCM_16")

    total = len(list(config.POSITIVE_DIR.glob("*.wav")))
    print(f"\n✓ Done – {total} positive clips in {config.POSITIVE_DIR}")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1 – Generate positive 'bumblebee' clips")
    print("=" * 60)

    voice_paths = download_voices()
    synthesise_clips(voice_paths)

    print("\nNext: run  step2_get_negatives.py")
