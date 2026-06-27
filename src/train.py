# =============================================================================
# train.py
# Training loop for the Early-Fusion Multimodal CNN.
#
# Runs:
#   1. Data loading and thermal-acoustic pairing
#   2. Train/test split (stratified, 80/20)
#   3. Spectrogram normalisation stats from training split only
#   4. Dataset and DataLoader construction
#   5. Model, loss, optimiser, and scheduler initialisation
#   6. Training loop with mixed-precision (AMP) and cosine annealing LR
#   7. Best checkpoint saved based on validation macro F1-score
#   8. Training curves (loss, accuracy, F1) saved to RESULTS_DIR
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from config import (
    DEVICE, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, RANDOM_STATE, TEST_SIZE,
    THERMAL_DIR, AUDIO_DIR, CHECKPOINT_DIR, RESULTS_DIR
)
from dataset import (
    load_thermal_paths, load_acoustic_segments,
    pair_thermal_acoustic, compute_spectrogram_stats,
    build_transforms, MultimodalDataset
)
from model import EarlyFusionNet
from visualize import plot_training_curves

# Note on mixed-precision import:
# torch.amp is the current path as of PyTorch 2.x.
# If you are on PyTorch 1.x, change to: from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler


def main():
    print(f"Device: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\nLoading thermal images...")
    thermal_paths, thermal_labels = load_thermal_paths(THERMAL_DIR)
    print(f"  {len(thermal_paths)} thermal images loaded")

    print("Loading and segmenting acoustic recordings...")
    acoustic_waves, acoustic_labels = load_acoustic_segments(AUDIO_DIR)
    print(f"  {len(acoustic_waves)} acoustic segments loaded")

    # ── 2. Pair thermal images with acoustic segments ─────────────────────────
    print("Pairing thermal and acoustic data by class...")
    paired_thermal, paired_acoustic, paired_labels = pair_thermal_acoustic(
        thermal_paths, thermal_labels,
        acoustic_waves, acoustic_labels
    )
    print(f"  {len(paired_thermal)} paired multimodal samples")

    # ── 3. Stratified train / test split ──────────────────────────────────────
    # Stratification ensures class proportions are preserved in both splits.
    (train_thermal, test_thermal,
     train_acoustic, test_acoustic,
     train_labels, test_labels) = train_test_split(
        paired_thermal, paired_acoustic, paired_labels,
        test_size=TEST_SIZE,
        stratify=paired_labels,
        random_state=RANDOM_STATE
    )
    print(f"  Train: {len(train_thermal)} samples | Test: {len(test_thermal)} samples")

    # ── 4. Compute spectrogram normalisation from training split only ─────────
    # Computing stats after the split prevents any test-set information
    # from leaking into the normalisation parameters.
    print("Computing spectrogram normalisation statistics (training split only)...")
    spec_mean, spec_std = compute_spectrogram_stats(train_acoustic)
    print(f"  Log-mel spectrogram — mean: {spec_mean:.4f}, std: {spec_std:.4f}")

    # ── 5. Build transforms and datasets ──────────────────────────────────────
    train_tf, test_tf = build_transforms(spec_mean, spec_std)

    train_dataset = MultimodalDataset(train_thermal, train_acoustic, train_labels, train_tf)
    test_dataset  = MultimodalDataset(test_thermal,  test_acoustic,  test_labels,  test_tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── 6. Model, loss, optimiser, scheduler ──────────────────────────────────
    model = EarlyFusionNet().to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class-weighted cross-entropy to handle any class imbalance
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Cosine annealing with warm restarts — periodically resets LR to escape
    # local minima; T_0 is the number of epochs in the first restart cycle.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    # Mixed-precision scaler — reduces memory and speeds up training on GPU.
    # Has no effect on CPU (ops run in float32 transparently).
    use_amp = DEVICE.type == "cuda"
    scaler  = GradScaler(device=DEVICE.type, enabled=use_amp)

    # ── 7. Training loop ───────────────────────────────────────────────────────
    best_f1 = 0.0
    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []
    val_f1s                    = []

    print(f"\nTraining for {MAX_EPOCHS} epochs...\n{'─'*60}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = correct = total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type, enabled=use_amp):
                logits = model(x)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            correct      += (logits.argmax(1) == y).sum().item()
            total        += y.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(correct / total)
        scheduler.step()

        # ── Validate (on held-out test set) ────────────────────────────────
        # Note: the test set is used for both validation and final evaluation.
        # Best checkpoint selection is therefore optimistic; reported metrics
        # should be interpreted accordingly.
        model.eval()
        val_loss = val_correct = val_total = 0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for x, y in test_loader:
                logits    = model(x.to(DEVICE))
                val_loss += criterion(logits, y.to(DEVICE)).item()
                pred      = logits.argmax(1)
                val_correct += (pred == y.to(DEVICE)).sum().item()
                val_total   += y.size(0)
                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(y.numpy())

        val_losses.append(val_loss / len(test_loader))
        val_accs.append(val_correct / val_total)

        f1 = classification_report(
            val_trues, val_preds, output_dict=True, zero_division=0
        )["macro avg"]["f1-score"]
        val_f1s.append(f1)

        print(
            f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
            f"Train loss: {train_losses[-1]:.4f} acc: {train_accs[-1]:.4f} | "
            f"Val loss: {val_losses[-1]:.4f} acc: {val_accs[-1]:.4f} | "
            f"Macro F1: {f1:.4f}"
        )

        # Save best checkpoint based on validation macro F1
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_fusion.pth")
            )
            print(f"  ✓ New best macro F1: {best_f1:.4f} — checkpoint saved")

    print(f"\nTraining complete. Best macro F1: {best_f1:.4f}")

    # ── 8. Save training curves ────────────────────────────────────────────────
    plot_training_curves(
        train_losses, val_losses,
        train_accs,   val_accs,
        val_f1s,
        save_dir=RESULTS_DIR
    )
    print(f"Training curves saved to {RESULTS_DIR}/")

    # ── Return test split objects for evaluation ───────────────────────────────
    # These in-memory objects are returned so main.py can pass them directly
    # to evaluate.py without reloading data from disk.
    return dict(
        test_thermal=test_thermal,
        test_acoustic=test_acoustic,
        test_labels=test_labels,
        test_loader=test_loader,
        test_tf=test_tf,
    )


if __name__ == "__main__":
    main()
