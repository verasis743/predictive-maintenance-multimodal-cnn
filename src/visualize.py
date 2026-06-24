# =============================================================================
# visualize.py
# Plotting utilities for training diagnostics and evaluation outputs.
#
# Functions:
#   plot_training_curves   — loss, accuracy, and F1 vs epoch
#   plot_confusion_matrix  — heatmap with class labels
#   plot_umap              — UMAP feature space visualisation
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP plots will be skipped.")
    print("Install with: pip install umap-learn")

from config import CLASS_NAMES

# ── Shared plot style constants ────────────────────────────────────────────────
_LABEL_FONT  = 13
_TICK_FONT   = 12
_LEGEND_FONT = 11
_GRID_STYLE  = dict(linestyle="--", alpha=0.5, color="gray", linewidth=0.7)
_COLORS      = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue, orange, green


# ── Training curves ────────────────────────────────────────────────────────────

def plot_training_curves(
        train_losses: list, val_losses: list,
        train_accs:   list, val_accs:   list,
        val_f1s:      list,
        save_dir:     str = "results") -> None:
    """
    Save three training diagnostic plots to save_dir:
      1. Training vs validation loss
      2. Training vs validation accuracy
      3. Validation macro F1-score

    Args:
        train_losses: Per-epoch training loss values.
        val_losses:   Per-epoch validation loss values.
        train_accs:   Per-epoch training accuracy values.
        val_accs:     Per-epoch validation accuracy values.
        val_f1s:      Per-epoch validation macro F1 values.
        save_dir:     Directory where PNG files are saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses,   label="Val loss")
    ax.set_title("Training and Validation Loss", fontsize=_LABEL_FONT + 1)
    ax.set_xlabel("Epoch", fontsize=_LABEL_FONT)
    ax.set_ylabel("Loss",  fontsize=_LABEL_FONT)
    ax.tick_params(labelsize=_TICK_FONT)
    ax.legend(fontsize=_LEGEND_FONT)
    ax.grid(**_GRID_STYLE)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_accs, label="Train accuracy", color="green")
    ax.plot(epochs, val_accs,   label="Val accuracy",   color="red")
    ax.set_title("Training and Validation Accuracy", fontsize=_LABEL_FONT + 1)
    ax.set_xlabel("Epoch",    fontsize=_LABEL_FONT)
    ax.set_ylabel("Accuracy", fontsize=_LABEL_FONT)
    ax.tick_params(labelsize=_TICK_FONT)
    ax.legend(fontsize=_LEGEND_FONT)
    ax.grid(**_GRID_STYLE)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close(fig)

    # F1 curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_f1s, label="Val macro F1", color="purple")
    ax.set_title("Validation Macro F1-Score", fontsize=_LABEL_FONT + 1)
    ax.set_xlabel("Epoch",      fontsize=_LABEL_FONT)
    ax.set_ylabel("Macro F1",   fontsize=_LABEL_FONT)
    ax.tick_params(labelsize=_TICK_FONT)
    ax.legend(fontsize=_LEGEND_FONT)
    ax.grid(**_GRID_STYLE)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "val_f1_curve.png"), dpi=300)
    plt.close(fig)

    print(f"Training curves saved: loss_curve.png, accuracy_curve.png, val_f1_curve.png")


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
        cm:        np.ndarray,
        title:     str = "Confusion Matrix",
        save_path: str = "results/confusion_matrix.png") -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm:        Confusion matrix array from sklearn.metrics.confusion_matrix.
        title:     Plot title string.
        save_path: Full path (including filename) to save the PNG.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    ax.set_title(title,       fontsize=_LABEL_FONT + 1)
    ax.set_xlabel("Predicted", fontsize=_LABEL_FONT)
    ax.set_ylabel("True",      fontsize=_LABEL_FONT)
    ax.tick_params(labelsize=_TICK_FONT)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix saved: {save_path}")


# ── UMAP feature visualisation ─────────────────────────────────────────────────

def plot_umap(
        model=None,
        data_loader: DataLoader = None,
        device: torch.device    = None,
        title:     str = "UMAP Feature Visualisation",
        save_path: str = "results/umap.png",
        features:  np.ndarray = None,
        labels:    np.ndarray = None,
        precomputed: bool     = False) -> None:
    """
    Generate and save a UMAP 2D projection of model feature representations.

    Can be called in two modes:
      1. Live extraction (precomputed=False): pass model + data_loader + device.
         Features are extracted using model.extract_features() before UMAP.
      2. Pre-extracted (precomputed=True): pass features + labels arrays directly.
         Used when features were already extracted during noise evaluation.

    Args:
        model:       EarlyFusionNet with extract_features() method.
        data_loader: DataLoader yielding (x, y) batches.
        device:      Torch device.
        title:       Plot title string.
        save_path:   Full path (including filename) to save the PNG.
        features:    Pre-extracted feature array [N, D] (precomputed mode).
        labels:      Ground-truth label array [N] (precomputed mode).
        precomputed: If True, use features/labels directly without extraction.
    """
    if not UMAP_AVAILABLE:
        print("Skipping UMAP plot — umap-learn not installed.")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # ── Extract features if not pre-computed ──────────────────────────────
    if not precomputed:
        if model is None or data_loader is None or device is None:
            raise ValueError(
                "model, data_loader, and device are required when precomputed=False."
            )
        model.eval()
        feat_list, label_list = [], []
        with torch.no_grad():
            for x, y in data_loader:
                f = model.extract_features(x.to(device))
                feat_list.append(f.cpu().numpy().reshape(x.shape[0], -1))
                label_list.append(y.numpy())
        features = np.concatenate(feat_list)
        labels   = np.concatenate(label_list)

    if features.shape[0] == 0:
        print(f"Warning: No features available for UMAP — skipping {save_path}")
        return

    # ── Fit UMAP ──────────────────────────────────────────────────────────
    reducer   = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=42)
    embedding = reducer.fit_transform(features)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(CLASS_NAMES):
        mask = labels == i
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            label=cls,
            c=_COLORS[i],
            alpha=0.7,
            s=60,
            edgecolors="k",
            linewidth=0.3
        )
    ax.set_title(title,    fontsize=_LABEL_FONT + 1)
    ax.set_xlabel("UMAP 1", fontsize=_LABEL_FONT)
    ax.set_ylabel("UMAP 2", fontsize=_LABEL_FONT)
    ax.tick_params(labelsize=_TICK_FONT, direction="in", length=4)
    ax.legend(fontsize=_LEGEND_FONT, frameon=True, edgecolor="black")
    ax.grid(**_GRID_STYLE)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"UMAP plot saved: {save_path}")
