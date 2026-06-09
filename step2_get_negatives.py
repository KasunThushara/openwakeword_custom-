"""
step2_get_negatives.py  (REWRITTEN — TTS-only negatives)
══════════════════════════════════════════════════════════

ROOT CAUSE FIX
══════════════
The previous version used LibriSpeech (real recorded speech) as negatives
while positives were Piper TTS. The Google embedding backbone puts TTS and
real speech in DIFFERENT regions of embedding space. The classifier learned:
  "TTS-style embeddings  →  bumblebee = 1"
  "Real-speech embeddings  →  bumblebee = 0"

This broke live inference because your real voice → 0 regardless of what
you said.

The fix: generate negatives using the SAME Piper TTS voices as the positives,
but with many different words and phrases. Now both classes look like TTS in
the embedding space, so the classifier is forced to learn the actual word
"bumblebee" rather than the recording style.
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

TARGET_SECONDS = 1.5
TARGET_SAMPLES = int(config.SAMPLE_RATE * TARGET_SECONDS)

# ── Negative phrases: diverse English covering many phonemes ──────
# These must NOT contain "bumblebee" or "bumble bee".
# They cover common home-assistant utterances, random speech, numbers,
# and phonemes that partially overlap with "bumblebee" to train harder.
NEGATIVE_PHRASES = [
    # Similar-sounding words (hardest negatives)
    "bumble", "humble", "stumble", "crumble", "tumble",
    "rumble", "mumble", "grumble", "jumble", "fumble",
    "bumpy ride", "bubble tea", "lumber yard", "number three",
    "come on baby", "bumper car", "something funny",

    # Common home assistant commands
    "turn on the lights", "turn off the lights",
    "set a timer for five minutes", "play some music",
    "what is the weather today", "call mom",
    "add milk to the shopping list", "set an alarm",
    "volume up", "volume down", "pause the music",
    "next song", "skip this", "stop",
    "turn on the fan", "turn off the tv",
    "open the garage", "lock the door",
    "dim the lights", "brighten the lights",

    # Numbers and letters
    "one two three four five",
    "six seven eight nine ten",
    "alpha bravo charlie delta",
    "a b c d e f g",

    # Common words and short phrases
    "hello", "goodbye", "yes", "no", "please", "thank you",
    "good morning", "good night", "good afternoon",
    "how are you", "fine thank you",
    "the quick brown fox jumps over the lazy dog",
    "to be or not to be that is the question",
    "all systems are operational",
    "the temperature is twenty degrees",
    "it is currently cloudy with a chance of rain",
    "your package has been delivered",
    "meeting starts in ten minutes",
    "battery level is at eighty percent",
    "connection established",
    "downloading update",
    "search results for your query",
    "playing next track",
    "lights are now off",
    "alarm set for seven am",

    # Phonetically diverse single words
    "elephant", "crocodile", "refrigerator", "university",
    "computer", "telephone", "microphone", "amplifier",
    "knowledge", "adventure", "tomorrow", "yesterday",
    "together", "whenever", "whatever", "somewhere",
    "everything", "something", "anything", "nothing",
    "basketball", "waterfall", "thunderstorm", "underground",
]


def pad_to_target(audio: np.ndarray) -> np.ndarray:
    """Same padding as step1: surround with ambient noise to reach 1.5s."""
    n = len(audio)
    if n >= TARGET_SAMPLES:
        start = (n - TARGET_SAMPLES) // 2
        return audio[start : start + TARGET_SAMPLES].copy()
    gap  = TARGET_SAMPLES - n
    pre  = random.randint(gap // 4, 3 * gap // 4)
    post = gap - pre
    amp  = max(float(np.abs(audio).mean()) * random.uniform(0.02, 0.08), 5e-4)
    return np.concatenate([
        (np.random.randn(pre)  * amp).astype(np.float32),
        audio.astype(np.float32),
        (np.random.randn(post) * amp).astype(np.float32),
    ])


def generate_tts_negatives() -> None:
    """Generate negative clips using Piper TTS (same voices as step1)."""
    try:
        from piper.voice import PiperVoice
    except ImportError:
        raise SystemExit("✗ pip install piper-tts")

    voice_files = sorted(config.VOICES_DIR.glob("*.onnx"))
    if not voice_files:
        raise SystemExit("✗ No voices found in voices/  — run step1 first.")

    print(f"\n[Step 2] Generating {config.N_NEGATIVE_CLIPS} TTS negative clips …")
    print(f"  Voices : {[v.stem for v in voice_files]}")
    print(f"  Phrases: {len(NEGATIVE_PHRASES)} different phrases")
    print(f"  Output : {config.NEGATIVE_DIR}\n")

    existing = list(config.NEGATIVE_DIR.glob("neg_*.wav"))
    start    = len(existing)
    if start >= config.N_NEGATIVE_CLIPS:
        print(f"  ↷ Already have {start} negatives — skipping.")
        return

    voices = [PiperVoice.load(str(p)) for p in voice_files]

    for i in tqdm(range(start, config.N_NEGATIVE_CLIPS), unit="clip"):
        voice  = random.choice(voices)
        phrase = random.choice(NEGATIVE_PHRASES)

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

        audio = pad_to_target(audio)

        out = config.NEGATIVE_DIR / f"neg_{i:06d}.wav"
        sf.write(str(out), audio, config.SAMPLE_RATE, subtype="PCM_16")

    print(f"\n✓ Done – {config.N_NEGATIVE_CLIPS} negative clips in {config.NEGATIVE_DIR}")


if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 2 – Generate TTS negative clips  (REWRITTEN)")
    print("=" * 60)
    print("""
  KEY CHANGE: negatives are now generated with the SAME Piper TTS
  voices as the positive clips.  Both classes now live in the same
  TTS embedding region, so the classifier must learn the actual word
  'bumblebee' rather than 'is this TTS or real speech?'
""")

    # Remove any old LibriSpeech negatives (different embedding style)
    old_negs = list(config.NEGATIVE_DIR.glob("neg_*.wav"))
    if old_negs:
        print(f"  Removing {len(old_negs)} old negatives …")
        for f in old_negs:
            f.unlink()
    old_adv = list(config.NEGATIVE_DIR.glob("adv_*.wav"))
    if old_adv:
        print(f"  Removing {len(old_adv)} old adversarial negatives …")
        for f in old_adv:
            f.unlink()

    generate_tts_negatives()
    print("\nNext: delete features/  then run step3_extract_features.py")