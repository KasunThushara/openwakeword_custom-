"""
step4_train.py
══════════════
Trains a small fully-connected neural network on the pre-computed
openWakeWord embeddings, then exports it as  model/bumblebee.onnx

Architecture
────────────
Input  (batch, 16, 96)   →  Flatten  →  Linear(1536, 128)  →  ReLU
       →  Dropout(0.3)   →  Linear(128, 64)  →  ReLU
       →  Linear(64, 1)  →  Sigmoid
Output (batch, 1)   — score between 0 (not bumblebee) and 1 (bumblebee)

Expected run time on CPU:  ~5–15 minutes
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config


# ═══════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════
class WakeWordModel(nn.Module):
    """
    Small classifier that takes a sequence of WINDOW_FRAMES × EMBEDDING_DIM
    embeddings and predicts a single wake-word score.
    """
    def __init__(
        self,
        window_frames: int = config.WINDOW_FRAMES,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_size:   int = config.HIDDEN_SIZE,
    ):
        super().__init__()
        input_size = window_frames * embedding_dim   # 16 × 96 = 1 536

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # NO SIGMOID HERE – BCEWithLogitsLoss applies sigmoid internally
            # Sigmoid will be added during ONNX export for inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, WINDOW_FRAMES, EMBEDDING_DIM)
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════
def build_datasets():
    """Load .npy feature files, combine, shuffle, return DataLoaders."""
    print("\n  Loading feature files …")
    pos_tr = np.load(config.FEATURES_DIR / "positive_train.npy")
    pos_va = np.load(config.FEATURES_DIR / "positive_val.npy")
    neg_tr = np.load(config.FEATURES_DIR / "negative_train.npy")
    neg_va = np.load(config.FEATURES_DIR / "negative_val.npy")

    # Balance: cap negatives at 3× positives so classes aren't wildly skewed
    max_neg_train = min(len(neg_tr), len(pos_tr) * 3)
    max_neg_val   = min(len(neg_va), len(pos_va) * 3)
    neg_tr = neg_tr[np.random.choice(len(neg_tr), max_neg_train, replace=False)]
    neg_va = neg_va[np.random.choice(len(neg_va), max_neg_val,   replace=False)]

    X_train = np.concatenate([pos_tr, neg_tr], axis=0).astype(np.float32)
    y_train = np.array([1.0] * len(pos_tr) + [0.0] * len(neg_tr), dtype=np.float32)
    X_val   = np.concatenate([pos_va, neg_va], axis=0).astype(np.float32)
    y_val   = np.array([1.0] * len(pos_va) + [0.0] * len(neg_va), dtype=np.float32)

    # Shuffle
    rng = np.random.default_rng(42)
    tr_idx = rng.permutation(len(X_train))
    va_idx = rng.permutation(len(X_val))
    X_train, y_train = X_train[tr_idx], y_train[tr_idx]
    X_val,   y_val   = X_val[va_idx],   y_val[va_idx]

    print(f"  Train  pos={len(pos_tr):4d}  neg={len(neg_tr):4d}  total={len(X_train):4d}")
    print(f"  Val    pos={len(pos_va):4d}  neg={len(neg_va):4d}  total={len(X_val):4d}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    return train_dl, val_dl


def train(model: WakeWordModel, train_dl, val_dl) -> WakeWordModel:
    device    = torch.device("cpu")
    model     = model.to(device)

    # Weighted BCE: give positive examples extra weight since the model
    # will encounter far more negative audio in real use.
    pos_weight = torch.tensor([3.0])   # tune if you get too many false positives
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state    = None

    print(f"\n  Training for {config.EPOCHS} epochs …")
    for epoch in range(1, config.EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for X_batch, y_batch in train_dl:
            X_batch = X_batch.to(device)
            y_batch = y_batch.unsqueeze(1).to(device)

            optimizer.zero_grad()
            # Forward pass: model outputs raw logits (no sigmoid)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * len(X_batch)
            preds    = (torch.sigmoid(logits) >= 0.5).float()
            tr_correct += (preds == y_batch).sum().item()
            tr_total   += len(X_batch)

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        va_loss = va_correct = va_tp = va_fp = va_fn = va_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch = X_batch.to(device)
                y_batch = y_batch.unsqueeze(1).to(device)
                logits  = model(X_batch)
                loss    = criterion(logits, y_batch)
                va_loss += loss.item() * len(X_batch)
                preds    = (torch.sigmoid(logits) >= 0.5).float()
                va_correct += (preds == y_batch).sum().item()
                va_tp += ((preds == 1) & (y_batch == 1)).sum().item()
                va_fp += ((preds == 1) & (y_batch == 0)).sum().item()
                va_fn += ((preds == 0) & (y_batch == 1)).sum().item()
                va_total   += len(X_batch)

        tr_loss_avg = tr_loss / tr_total
        va_loss_avg = va_loss / va_total
        tr_acc = 100 * tr_correct / tr_total
        va_acc = 100 * va_correct / va_total
        recall    = va_tp / max(va_tp + va_fn, 1)
        precision = va_tp / max(va_tp + va_fp, 1)

        scheduler.step()

        # Save best model
        if va_loss_avg < best_val_loss:
            best_val_loss = va_loss_avg
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ← best"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{config.EPOCHS}"
                f"  tr_loss={tr_loss_avg:.4f}  va_loss={va_loss_avg:.4f}"
                f"  tr_acc={tr_acc:.1f}%  va_acc={va_acc:.1f}%"
                f"  recall={recall:.3f}  prec={precision:.3f}"
                f"{marker}"
            )

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"\n  ✓ Training done.  Best val loss: {best_val_loss:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  ONNX export
# ═══════════════════════════════════════════════════════════════════
def export_onnx(model: WakeWordModel) -> Path:
    """
    Export the trained model to ONNX with Sigmoid applied for inference.
    Input shape:  (batch, WINDOW_FRAMES, EMBEDDING_DIM)
    Output shape: (batch, 1)  — sigmoid-applied scores [0, 1]
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("✗ Install:  pip install onnx onnxruntime")

    model.eval()
    out_path = config.MODEL_DIR / f"{config.MODEL_NAME}.onnx"

    # Wrap the model with Sigmoid for export
    class ModelWithSigmoid(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
        def forward(self, x):
            return torch.sigmoid(self.base(x))

    model_with_sig = ModelWithSigmoid(model)

    dummy = torch.randn(1, config.WINDOW_FRAMES, config.EMBEDDING_DIM)

    torch.onnx.export(
        model_with_sig,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Verify the exported model loads cleanly
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    # Quick inference test
    sess = ort.InferenceSession(str(out_path))
    test_input = np.random.randn(1, config.WINDOW_FRAMES, config.EMBEDDING_DIM).astype(np.float32)
    result = sess.run(None, {"input": test_input})
    score  = float(result[0][0][0])

    print(f"\n  ✓ ONNX model saved: {out_path}")
    print(f"    Input  shape : {sess.get_inputs()[0].shape}")
    print(f"    Output shape : {sess.get_outputs()[0].shape}")
    print(f"    Test score   : {score:.4f}  (random noise → should be near 0.5)")

    return out_path


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 4 – Train classifier + export ONNX")
    print("=" * 60)

    # Check features exist
    needed = ["positive_train.npy", "positive_val.npy",
              "negative_train.npy", "negative_val.npy"]
    for f in needed:
        if not (config.FEATURES_DIR / f).exists():
            raise SystemExit(
                f"✗ Missing: features/{f}\n"
                "  Run step3_extract_features.py first."
            )

    np.random.seed(42)
    torch.manual_seed(42)

    train_dl, val_dl = build_datasets()

    model = WakeWordModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_params:,}")

    model = train(model, train_dl, val_dl)
    onnx_path = export_onnx(model)

    print(f"\n✓ Done!  Model ready at: {onnx_path}")
    print("\nNext: run  step5_test.py  to test with your microphone")
