"""
step2_get_negatives.py
═══════════════════════
Downloads negative (non-wake-word) speech data from LibriSpeech
via HuggingFace datasets in STREAMING mode.

– No full dataset download required (saves ~6 GB)
– Streams only what we need, clips to 3-second chunks at 16 kHz
– Also generates adversarial negatives: words that sound like
  "bumblebee" but aren't (humble, crumble, stumble …)

Expected run time:  ~10–20 minutes  (depends on internet speed)
Disk used:          ~500 MB for 3 000 clips
"""

import io
import random
import wave
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

import config

# 3-second chunks cut from LibriSpeech utterances
CHUNK_SECONDS = 3


# ── Part A: stream LibriSpeech clean validation set ───────────────
def download_librispeech_negatives() -> None:
    print("\n[Step 2-A] Streaming LibriSpeech negatives …")
    print(f"  Target: {config.N_NEGATIVE_CLIPS} clips  →  {config.NEGATIVE_DIR}")

    try:
        from datasets import Audio, Features, Value, load_dataset
    except ImportError:
        raise SystemExit("✗ Install datasets:  pip install datasets")

    existing = list(config.NEGATIVE_DIR.glob("neg_*.wav"))
    if len(existing) >= config.N_NEGATIVE_CLIPS:
        print(f"  ↷ Already have {len(existing)} negative clips – skipping download.")
        return

    start = len(existing)
    count = start

    # librispeech validation-clean is ~340 MB of audio, streamed on demand
    features = Features(
        {
            "file": Value("string"),
            "audio": Audio(sampling_rate=None, decode=False),
            "text": Value("string"),
            "speaker_id": Value("int64"),
            "chapter_id": Value("int64"),
            "id": Value("string"),
        }
    )

    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="validation",
        streaming=True,
        features=features,
    )

    chunk_samples = config.SAMPLE_RATE * CHUNK_SECONDS
    pbar = tqdm(total=config.N_NEGATIVE_CLIPS - start, unit="clip")

    for example in ds:
        if count >= config.N_NEGATIVE_CLIPS:
            break

        audio_data = example["audio"]
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data["sampling_rate"]
        elif isinstance(audio_data, dict) and "bytes" in audio_data:
            buf = io.BytesIO(audio_data["bytes"])
            audio, sr = sf.read(buf)
            audio = audio.astype(np.float32)
        else:
            raise RuntimeError("Unsupported audio format from LibriSpeech stream")

        if sr != config.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

        # Split the utterance into 3-second chunks
        for j in range(len(audio) // chunk_samples):
            if count >= config.N_NEGATIVE_CLIPS:
                break
            chunk = audio[j * chunk_samples : (j + 1) * chunk_samples]

            out = config.NEGATIVE_DIR / f"neg_{count:06d}.wav"
            sf.write(str(out), chunk, config.SAMPLE_RATE, subtype="PCM_16")
            count += 1
            pbar.update(1)

    pbar.close()
    print(f"  ✓ Saved {count} negative clips")


# ── Part B: adversarial negatives (similar-sounding words) ────────
# These are the words most likely to cause false positives because
# they share phonemes with "bumblebee".  Generating them explicitly
# and labelling as NEGATIVE forces the model to discriminate.
ADVERSARIAL_PHRASES = [
    "bumble",
    "humble bee",
    "stumble",
    "crumble",
    "tumble",
    "rumble",
    "mumble",
    "grumble",
    "jumble",
    "bumpy",
    "bubble",
    "bumper",
    "lumber",
    "number three",
    "come on baby",          # shares the /b/ /iː/ pattern
]

N_ADVERSARIAL = 300    # 300 clips, ~20 per phrase


def generate_adversarial_negatives() -> None:
    """Use piper-tts to generate clips of similar-sounding words."""
    try:
        from piper.voice import PiperVoice
    except ImportError:
        print("  ⚠ piper-tts not available – skipping adversarial negatives.")
        return

    voice_files = list(config.VOICES_DIR.glob("*.onnx"))
    if not voice_files:
        print("  ⚠ No Piper voices found in voices/ – run step1 first.")
        return

    print(f"\n[Step 2-B] Generating {N_ADVERSARIAL} adversarial negative clips …")
    voices = [PiperVoice.load(str(p)) for p in voice_files]

    existing_adv = list(config.NEGATIVE_DIR.glob("adv_*.wav"))
    start = len(existing_adv)
    if start >= N_ADVERSARIAL:
        print(f"  ↷ Already have {start} adversarial clips – skipping.")
        return

    for i in tqdm(range(start, N_ADVERSARIAL), unit="clip"):
        voice  = random.choice(voices)
        phrase = random.choice(ADVERSARIAL_PHRASES)

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

        out = config.NEGATIVE_DIR / f"adv_{i:05d}.wav"
        sf.write(str(out), audio, config.SAMPLE_RATE, subtype="PCM_16")

    total_neg = len(list(config.NEGATIVE_DIR.glob("*.wav")))
    print(f"  ✓ Adversarial negatives done. Total negatives: {total_neg}")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 2 – Download / generate negative data")
    print("=" * 60)

    download_librispeech_negatives()
    generate_adversarial_negatives()

    total = len(list(config.NEGATIVE_DIR.glob("*.wav")))
    print(f"\n✓ Done – {total} total negative clips in {config.NEGATIVE_DIR}")
    print("\nNext: run  step3_extract_features.py")
