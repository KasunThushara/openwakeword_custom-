"""
step1_generate_clips.py  (FIXED)
═════════════════════════════════
Key fix: each positive clip is padded to exactly 1.5 seconds with low-level
ambient noise, with the wake word placed randomly in the centre third.

This ensures the embedding has ~18 real frames (not zero-padded short clips),
which matches what the model sees during live inference.
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

# ── Target clip length ─────────────────────────────────────────────
# Must be long enough to produce at least WINDOW_FRAMES embeddings.
# 1.5 s → ~18 frames at 80 ms/frame  (we only need 16)
TARGET_SECONDS = 1.5
TARGET_SAMPLES = int(config.SAMPLE_RATE * TARGET_SECONDS)   # 24 000

# ── Augmentation ───────────────────────────────────────────────────
augment = A.Compose([
    A.PitchShift(min_semitones=-3, max_semitones=3, p=0.6),
    A.TimeStretch(min_rate=0.85, max_rate=1.20, p=0.6),
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    A.LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7000, p=0.2),
])


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  ✓ already exists: {dest.name}")
        return
    print(f"  ↓ downloading {dest.name} …")

    class _Progress:
        def __init__(self): self.bar = None
        def __call__(self, block, block_size, total):
            if self.bar is None:
                self.bar = tqdm(total=total, unit="B", unit_scale=True, leave=False)
            self.bar.update(block_size)

    urllib.request.urlretrieve(url, dest, reporthook=_Progress())
    print(f"  ✓ saved {dest.name}")


def pad_to_target(audio: np.ndarray) -> np.ndarray:
    """
    Pad a short TTS clip to TARGET_SAMPLES by surrounding it with
    low-level ambient noise.  The word is placed in a random position
    within the centre 50 % of the padded clip so the embedding window
    always captures it.

    Without this fix every positive clip is zero-padded → the model
    learns 'lots of zeros = bumblebee' instead of learning the word.
    """
    n = len(audio)
    if n >= TARGET_SAMPLES:
        # Trim to target from the centre (rare for short words)
        start = (n - TARGET_SAMPLES) // 2
        return audio[start : start + TARGET_SAMPLES].copy()

    gap = TARGET_SAMPLES - n
    # Place word randomly in the middle 50 % of the clip
    pre  = random.randint(gap // 4, 3 * gap // 4)
    post = gap - pre

    # Ambient noise floor: very quiet (SNR ≈ 30 dB above noise)
    noise_amp = float(np.abs(audio).mean()) * random.uniform(0.02, 0.08)
    noise_amp = max(noise_amp, 5e-4)   # minimum noise floor
    pre_noise  = (np.random.randn(pre)  * noise_amp).astype(np.float32)
    post_noise = (np.random.randn(post) * noise_amp).astype(np.float32)

    return np.concatenate([pre_noise, audio.astype(np.float32), post_noise])


def download_voices() -> list[Path]:
    onnx_paths = []
    print("\n[Step 1-A] Downloading Piper voice models …")
    for vm in config.VOICE_MODELS:
        onnx_path = config.VOICES_DIR / f"{vm['name']}.onnx"
        json_path = config.VOICES_DIR / f"{vm['name']}.onnx.json"
        download(vm["onnx"], onnx_path)
        download(vm["json"], json_path)
        onnx_paths.append(onnx_path)
    return onnx_paths


def synthesise_clips(voice_paths: list[Path]) -> None:
    try:
        from piper.voice import PiperVoice
    except ImportError:
        raise SystemExit("\n✗ pip install piper-tts\n")

    print(f"\n[Step 1-B] Loading {len(voice_paths)} Piper voice(s) …")
    voices = [PiperVoice.load(str(p)) for p in voice_paths]
    print("  ✓ voices loaded")

    existing   = list(config.POSITIVE_DIR.glob("*.wav"))
    start_idx  = len(existing)
    if start_idx > 0:
        print(f"  ↷ {start_idx} clips exist – resuming")

    print(f"\n[Step 1-B] Generating {config.N_POSITIVE_CLIPS} positive clips …")
    print(f"  Each clip padded to {TARGET_SECONDS}s  ({TARGET_SAMPLES} samples)")
    print(f"  Phrases: {config.PHRASES}\n")

    for i in tqdm(range(start_idx, config.N_POSITIVE_CLIPS), unit="clip"):
        voice  = random.choice(voices)
        phrase = random.choice(config.PHRASES)

        # ── Synthesise ──────────────────────────────────────────────
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(phrase, wf)

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            n_ch   = wf.getnchannels()
            src_sr = wf.getframerate()
            raw    = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch)[:, 0]
        if src_sr != config.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=src_sr, target_sr=config.SAMPLE_RATE)

        # ── Augment (pitch, speed, noise) ───────────────────────────
        audio = augment(samples=audio, sample_rate=config.SAMPLE_RATE)

        # ── THE KEY FIX: pad to 1.5 s with ambient noise ────────────
        audio = pad_to_target(audio)

        # ── Save as 16-bit 16 kHz WAV ────────────────────────────────
        out = config.POSITIVE_DIR / f"bumblebee_{i:05d}.wav"
        sf.write(str(out), audio, config.SAMPLE_RATE, subtype="PCM_16")

    total = len(list(config.POSITIVE_DIR.glob("*.wav")))
    print(f"\n✓ Done – {total} positive clips  (each {TARGET_SECONDS}s)")


if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1 – Generate positive 'bumblebee' clips  (FIXED)")
    print("=" * 60)
    voice_paths = download_voices()
    synthesise_clips(voice_paths)
    print("\nNext: DELETE features/  then run step3_extract_features.py")