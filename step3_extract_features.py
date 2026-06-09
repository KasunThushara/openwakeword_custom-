"""
step3_extract_features.py
══════════════════════════
Runs every WAV clip through the openWakeWord backbone model
(the frozen Google speech embedding network) and saves the resulting
feature arrays as .npy files.

This is the SLOW step – plan for 30–60 minutes on CPU.
The backbone runs only once; all subsequent training re-uses the .npy files.

Output files
────────────
features/positive_train.npy   shape (N_train, WINDOW_FRAMES, 96)
features/positive_val.npy     shape (N_val,   WINDOW_FRAMES, 96)
features/negative_train.npy   shape (M_train, WINDOW_FRAMES, 96)
features/negative_val.npy     shape (M_val,   WINDOW_FRAMES, 96)
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config

# ── Sanity-check openwakeword is installed ─────────────────────────
try:
    import openwakeword
    from openwakeword.utils import AudioFeatures
except ImportError:
    raise SystemExit(
        "✗ openwakeword not found.\n"
        "  Run:  pip install openwakeword\n"
        "        pip install torch==2.2.2  # re-pin torch if it got downgraded\n"
    )


# ── Helpers ────────────────────────────────────────────────────────
def load_wav_16k(path: Path) -> np.ndarray | None:
    """Load a WAV file as int16 mono at 16 kHz."""
    try:
        audio, sr = sf.read(str(path), dtype="int16")
    except Exception as e:
        print(f"  ⚠ Could not read {path.name}: {e}")
        return None

    if audio.ndim > 1:
        audio = audio[:, 0]          # keep left channel only

    # Basic sample-rate guard – clips should already be 16 kHz from step 1/2
    if sr != config.SAMPLE_RATE:
        import librosa
        audio_f = audio.astype(np.float32) / 32768.0
        audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        audio   = (audio_f * 32768).astype(np.int16)

    return audio


def clip_to_window(emb: np.ndarray, is_positive: bool) -> np.ndarray:
    """
    Select WINDOW_FRAMES consecutive frames from an embedding array.

    Positive clips: take the centre window (wake word is in the middle)
    Negative clips: take a random window
    Shape in:  (n_frames, 96)
    Shape out: (WINDOW_FRAMES, 96)
    """
    n_frames = emb.shape[0]
    W = config.WINDOW_FRAMES

    if n_frames >= W:
        if is_positive:
            start = max(0, (n_frames - W) // 2)
        else:
            start = np.random.randint(0, n_frames - W + 1)
        return emb[start : start + W]
    else:
        # Clip shorter than window – pad with zeros on the right
        pad = W - n_frames
        return np.pad(emb, ((0, pad), (0, 0)), mode="constant")


def embed_batch(paths: list[Path], is_positive: bool, F: "AudioFeatures", desc: str) -> np.ndarray:
    """
    Load a list of WAV files, embed them, and return windows.
    Returns:  np.ndarray  shape (len(paths), WINDOW_FRAMES, 96)
    """
    BATCH = 32    # process in batches to keep RAM usage low

    windows = []
    for i in tqdm(range(0, len(paths), BATCH), desc=desc, unit="batch"):
        batch_paths = paths[i : i + BATCH]
        clips = []
        valid_idx = []
        for j, p in enumerate(batch_paths):
            audio = load_wav_16k(p)
            if audio is not None:
                clips.append(audio)
                valid_idx.append(j)

        if not clips:
            continue

        # embed_clips expects a numpy array of shape (N, samples)
        # where every clip in the batch is the same length. The backbone
        # computes mel frames with: n_frames = ceil(samples/160 - 3).
        # To ensure all clips produce the same n_frames, pad/truncate
        # every clip to a target sample length corresponding to the
        # maximum n_frames in the batch.
        def _n_frames(samples: int) -> int:
            return int(np.ceil(samples / 160.0 - 3.0))

        max_frames = max(_n_frames(audio.shape[0]) for audio in clips)
        target_samples = int((max_frames + 3) * 160)

        batch = np.stack(
            [
                (np.pad(audio, (0, target_samples - audio.shape[0]), mode="constant")
                 if audio.shape[0] < target_samples else audio[:target_samples])
                for audio in clips
            ]
        )

        # embed_clips returns (n, n_frames, 96)
        embeddings = F.embed_clips(batch, batch_size=len(clips))

        for emb in embeddings:
            w = clip_to_window(emb, is_positive)
            windows.append(w)

    return np.array(windows, dtype=np.float32)   # (N, W, 96)


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 3 – Extract openWakeWord embeddings")
    print("=" * 60)

    # ── Check source clips exist ───────────────────────────────────
    pos_clips = sorted(config.POSITIVE_DIR.glob("*.wav"))
    neg_clips = sorted(config.NEGATIVE_DIR.glob("*.wav"))

    if len(pos_clips) < 100:
        raise SystemExit(
            f"✗ Only {len(pos_clips)} positive clips found.\n"
            "  Run step1_generate_clips.py first."
        )
    if len(neg_clips) < 100:
        raise SystemExit(
            f"✗ Only {len(neg_clips)} negative clips found.\n"
            "  Run step2_get_negatives.py first."
        )

    print(f"\n  Positive clips : {len(pos_clips)}")
    print(f"  Negative clips : {len(neg_clips)}")

    # ── Skip if features already exist ────────────────────────────
    expected = ["positive_train.npy", "positive_val.npy",
                "negative_train.npy", "negative_val.npy"]
    if all((config.FEATURES_DIR / f).exists() for f in expected):
        print("\n  ↷ Feature files already exist – delete features/ to re-compute.")
        sys.exit(0)

    # ── Ensure backbone models are downloaded ─────────────────────
    print("\n  Downloading openWakeWord backbone models (once only) …")
    openwakeword.utils.download_models()

    # ── Initialise AudioFeatures (loads backbone ONNX) ────────────
    print("\n  Loading AudioFeatures backbone …")
    F = AudioFeatures(device="cpu", ncpu=4)
    print("  ✓ Backbone loaded")

    # ── Train / val split ──────────────────────────────────────────
    pos_train_paths, pos_val_paths = train_test_split(
        pos_clips, test_size=config.VAL_SPLIT, random_state=42
    )
    neg_train_paths, neg_val_paths = train_test_split(
        neg_clips, test_size=config.VAL_SPLIT, random_state=42
    )

    print(f"\n  Positive  train={len(pos_train_paths)}  val={len(pos_val_paths)}")
    print(f"  Negative  train={len(neg_train_paths)}  val={len(neg_val_paths)}")
    print(f"\n  This will take ~30–60 min on CPU.  Go grab a coffee ☕")

    # ── Embed all clips ────────────────────────────────────────────
    pos_train_feat = embed_batch(pos_train_paths, is_positive=True,  F=F, desc="Positive train")
    pos_val_feat   = embed_batch(pos_val_paths,   is_positive=True,  F=F, desc="Positive val  ")
    neg_train_feat = embed_batch(neg_train_paths, is_positive=False, F=F, desc="Negative train")
    neg_val_feat   = embed_batch(neg_val_paths,   is_positive=False, F=F, desc="Negative val  ")

    # ── Save ───────────────────────────────────────────────────────
    print("\n  Saving feature arrays …")
    np.save(config.FEATURES_DIR / "positive_train.npy", pos_train_feat)
    np.save(config.FEATURES_DIR / "positive_val.npy",   pos_val_feat)
    np.save(config.FEATURES_DIR / "negative_train.npy", neg_train_feat)
    np.save(config.FEATURES_DIR / "negative_val.npy",   neg_val_feat)

    print(f"\n✓ Features saved to {config.FEATURES_DIR}")
    print(f"  positive_train : {pos_train_feat.shape}")
    print(f"  positive_val   : {pos_val_feat.shape}")
    print(f"  negative_train : {neg_train_feat.shape}")
    print(f"  negative_val   : {neg_val_feat.shape}")
    print("\nNext: run  step4_train.py")
