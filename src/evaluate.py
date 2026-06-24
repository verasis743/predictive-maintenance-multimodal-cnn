# =============================================================================
# evaluate.py
# Model evaluation under clean and noisy conditions.
#
# Two evaluation modes:
#   1. Clean evaluation: standard inference on the held-out test set.
#   2. Noise robustness: acoustic channel is corrupted with 5 real-world
#      noise types at 6 SNR levels (-10 to +20 dB), each run 3 times.
#      Mean ± std accuracy and macro F1 are reported per condition.
#
# Noise types evaluated:
#   - White Gaussian noise (generated programmatically)
#   - Factory floor noise   (factory.mp3)
#   - Cafeteria noise       (cafeteria.mp3)
#   - Traffic noise         (traffic.mp3)
#   - Rain noise            (rain.mp3)
#
# Note on noise audio files:
#   The .mp3 files are NOT included in this repository due to copyright
#   restrictions. See data/noise_sources/README.md for instructions on
#   downloading free equivalents from freesound.org.
# =============================================================================

import os
import numpy as np
import torch
import torchaudio
import cv2
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

from config import (
    DEVICE, CLASS_NAMES, SAMPLE_RATE, IMAGE_SIZE,
    SNR_LEVELS, NOISE_TYPES, EVAL_RUNS,
    NOISE_DIR, CHECKPOINT_DIR, RESULTS_DIR
)
from dataset import to_logmel
from model import EarlyFusionNet
from visualize import plot_confusion_matrix, plot_umap


# ── Noise loading ──────────────────────────────────────────────────────────────

def load_noise_sources(noise_dir: str, noise_types: list) -> dict:
    """
    Load real-world noise audio files and resample to SAMPLE_RATE.
    White Gaussian noise is generated on-the-fly (no file needed).

    Args:
        noise_dir:   Directory containing <noise_type>.mp3 files.
        noise_types: List of noise type names (must match filenames).

    Returns:
        Dict mapping noise_type -> waveform tensor [time] or None (white noise).

    Raises:
        FileNotFoundError if a required .mp3 file is missing.
        See data/noise_sources/README.md for download instructions.
    """
    noises = {}
    for name in noise_types:
        if name == "white":
            noises[name] = None  # generated programmatically in add_noise()
        else:
            path = os.path.join(noise_dir, f"{name}.mp3")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Noise file not found: {path}\n"
                    f"See data/noise_sources/README.md for download instructions."
                )
            wave, sr = torchaudio.load(path)
            wave = wave.mean(dim=0, keepdim=True)   # convert to mono
            if sr != SAMPLE_RATE:
                wave = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wave)
            noises[name] = wave.squeeze(0)           # [time]
    return noises


# ── SNR-controlled noise addition ─────────────────────────────────────────────

def add_noise(clean_wave: torch.Tensor,
              noise_wave: torch.Tensor | None,
              snr_db: float) -> torch.Tensor:
    """
    Add noise to a clean waveform at a specified Signal-to-Noise Ratio (SNR).

    The noise amplitude is scaled so that:
        SNR (dB) = 10 * log10(signal_power / noise_power)

    After mixing, the noisy waveform is peak-normalised to [-1, 1] to
    prevent clipping artifacts.

    Args:
        clean_wave: Clean mono waveform tensor [time].
        noise_wave: Noise waveform tensor [time], or None for white Gaussian noise.
        snr_db:     Target SNR in decibels.

    Returns:
        Noisy waveform tensor [1, time] (channel dimension added for compatibility
        with the log-mel transform pipeline).
    """
    clean_wave = clean_wave.clone()

    # Generate or tile noise to match clean signal length
    if noise_wave is None:
        noise_wave = torch.randn_like(clean_wave)   # white Gaussian noise
    else:
        repeat_times = (clean_wave.shape[0] // noise_wave.shape[0]) + 1
        noise_wave = noise_wave.repeat(repeat_times)[:clean_wave.shape[0]]

    signal_power = torch.mean(clean_wave ** 2)
    noise_power  = torch.mean(noise_wave ** 2)

    # Avoid division by zero for silent segments
    if signal_power == 0 or noise_power == 0:
        return clean_wave.unsqueeze(0)

    # Scale noise to achieve target SNR
    desired_noise_power = signal_power / (10 ** (snr_db / 10.0))
    scale      = torch.sqrt(desired_noise_power / noise_power)
    noisy_wave = clean_wave + scale * noise_wave

    # Peak normalise to prevent clipping
    max_val = noisy_wave.abs().max()
    if max_val > 0:
        noisy_wave = noisy_wave / max_val

    return noisy_wave.unsqueeze(0)   # [1, time]


# ── Clean evaluation ───────────────────────────────────────────────────────────

def evaluate_clean(model: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader) -> dict:
    """
    Evaluate the model on the clean (un-corrupted) test set.

    Args:
        model:       Trained EarlyFusionNet in eval mode.
        test_loader: DataLoader for the test set.

    Returns:
        Dict with keys: preds, trues, probs, accuracy, f1, report, cm.
    """
    model.eval()
    preds, trues, probs = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(DEVICE))
            probs.extend(logits.cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.numpy())

    acc    = accuracy_score(trues, preds)
    report = classification_report(
        trues, preds, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    cm     = confusion_matrix(trues, preds)
    f1     = report["macro avg"]["f1-score"]

    print("\n" + "=" * 60)
    print("CLEAN DATA EVALUATION")
    print("=" * 60)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    for cls in CLASS_NAMES:
        idx = CLASS_NAMES.index(cls)
        r   = report[str(idx)]
        print(f"{cls:<15} {r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['f1-score']:>8.4f} {r['support']:>8}")
    print(f"\nOverall accuracy : {acc:.4f}")
    print(f"Macro F1-score   : {f1:.4f}")

    return dict(preds=preds, trues=trues, probs=probs,
                accuracy=acc, f1=f1, report=report, cm=cm)


# ── Noise robustness evaluation ────────────────────────────────────────────────

def evaluate_noise_robustness(
        model:          torch.nn.Module,
        test_thermal:   list,
        test_acoustic:  list,
        test_labels:    list,
        noises:         dict,
        test_tf) -> dict:
    """
    Evaluate model robustness across noise types and SNR levels.

    For each (noise_type, SNR) combination:
      - The acoustic channel of every test sample is corrupted at the target SNR.
      - The thermal channel is unchanged (thermal imaging is noise-independent).
      - Inference is run EVAL_RUNS times; mean ± std accuracy and macro F1
        are recorded to account for stochastic white noise generation.

    Args:
        model:         Trained EarlyFusionNet in eval mode.
        test_thermal:  List of thermal image paths for the test split.
        test_acoustic: List of acoustic waveform tensors for the test split.
        test_labels:   Ground-truth class labels for the test split.
        noises:        Dict from load_noise_sources().
        test_tf:       Albumentations test transform (normalisation only).

    Returns:
        Nested dict: results[noise_type][snr_db] = {'acc': [...], 'f1': [...]}.
    """
    model.eval()
    results = {
        name: {snr: {"acc": [], "f1": []} for snr in SNR_LEVELS}
        for name in NOISE_TYPES
    }

    print("\n" + "=" * 60)
    print("NOISE ROBUSTNESS EVALUATION")
    print("=" * 60)

    for name in NOISE_TYPES:
        base_noise = noises[name]
        print(f"\nNoise type: {name.capitalize()}")

        for snr in SNR_LEVELS:
            for run in range(EVAL_RUNS):
                preds = []

                for idx in range(len(test_thermal)):
                    # Load and resize thermal image (unchanged — thermal is noise-free)
                    thermal = cv2.imread(test_thermal[idx])
                    thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
                    thermal = cv2.resize(thermal, (IMAGE_SIZE, IMAGE_SIZE))

                    # Corrupt acoustic channel at target SNR
                    clean_wave = test_acoustic[idx].squeeze(0)
                    noisy_wave = add_noise(clean_wave, base_noise, snr)

                    # Convert noisy waveform to log-mel spectrogram
                    spec    = to_logmel(noisy_wave).squeeze(0).cpu().numpy()
                    spec    = cv2.resize(spec, (IMAGE_SIZE, IMAGE_SIZE))
                    spec    = spec[:, :, np.newaxis]

                    # Fuse and infer
                    fused   = np.concatenate((thermal, spec), axis=2)
                    fused_t = test_tf(image=fused)["image"].unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        logits = model(fused_t)
                        preds.append(logits.argmax(1).item())

                acc = accuracy_score(test_labels, preds)
                f1  = classification_report(
                    test_labels, preds, output_dict=True, zero_division=0
                )["macro avg"]["f1-score"]

                results[name][snr]["acc"].append(acc)
                results[name][snr]["f1"].append(f1)

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MEAN ACCURACY AND MACRO F1 (± STD) PER NOISE TYPE AND SNR LEVEL")
    print("=" * 80)
    print(f"{'Noise':<12} {'SNR (dB)':<10} {'Accuracy (mean ± std)':<26} {'Macro F1 (mean ± std)'}")
    for name in NOISE_TYPES:
        for snr in SNR_LEVELS:
            acc_mean = np.mean(results[name][snr]["acc"])
            acc_std  = np.std(results[name][snr]["acc"])
            f1_mean  = np.mean(results[name][snr]["f1"])
            f1_std   = np.std(results[name][snr]["f1"])
            print(
                f"{name.capitalize():<12} {snr:<10} "
                f"{acc_mean:.4f} ± {acc_std:.4f}          "
                f"{f1_mean:.4f} ± {f1_std:.4f}"
            )

    return results


# ── Best / worst case analysis ─────────────────────────────────────────────────

def find_best_worst(results: dict) -> tuple:
    """
    Identify the noise × SNR condition with highest and lowest mean accuracy.

    Args:
        results: Output dict from evaluate_noise_robustness().

    Returns:
        Tuple of (best_info, worst_info) where each is
        (noise_name, snr_db, mean_accuracy).
    """
    best_acc,  best_info  = -1.0,  None
    worst_acc, worst_info =  2.0,  None

    for noise in results:
        for snr in results[noise]:
            acc_mean = np.mean(results[noise][snr]["acc"])
            if acc_mean > best_acc:
                best_acc  = acc_mean
                best_info = (noise, snr, acc_mean)
            if acc_mean < worst_acc:
                worst_acc  = acc_mean
                worst_info = (noise, snr, acc_mean)

    print(f"\nBest  condition: {best_info[0].capitalize()} noise "
          f"at {best_info[1]} dB → accuracy {best_acc:.4f}")
    print(f"Worst condition: {worst_info[0].capitalize()} noise "
          f"at {worst_info[1]} dB → accuracy {worst_acc:.4f}")

    return best_info, worst_info


# ── Main evaluation entry point ────────────────────────────────────────────────

def main(test_thermal, test_acoustic, test_labels, test_loader, test_tf):
    """
    Run full evaluation pipeline: clean + noise robustness + visualisations.

    Call this function from a notebook or script after training completes,
    passing the test split data and DataLoader constructed in train.py.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load best checkpoint
    model = EarlyFusionNet().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_fusion.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Clean evaluation ───────────────────────────────────────────────────
    clean_results = evaluate_clean(model, test_loader)

    plot_confusion_matrix(
        clean_results["cm"],
        title="Confusion Matrix — Clean Test Set",
        save_path=os.path.join(RESULTS_DIR, "confusion_clean.png")
    )

    plot_umap(
        model, test_loader, DEVICE,
        title="UMAP Feature Visualisation — Clean Test Set",
        save_path=os.path.join(RESULTS_DIR, "umap_clean.png")
    )

    # ── Noise robustness evaluation ────────────────────────────────────────
    noises = load_noise_sources(NOISE_DIR, NOISE_TYPES)

    noise_results = evaluate_noise_robustness(
        model, test_thermal, test_acoustic, test_labels, noises, test_tf
    )

    best_info, worst_info = find_best_worst(noise_results)

    # Confusion matrix and UMAP for best and worst noise conditions
    for tag, info in [("best", best_info), ("worst", worst_info)]:
        noise_name, snr, _ = info
        noise_wave = noises[noise_name]

        preds, all_features = [], []

        with torch.no_grad():
            for idx in range(len(test_thermal)):
                thermal = cv2.imread(test_thermal[idx])
                thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
                thermal = cv2.resize(thermal, (IMAGE_SIZE, IMAGE_SIZE))

                clean_wave = test_acoustic[idx].squeeze(0)
                noisy_wave = add_noise(clean_wave, noise_wave, snr)

                spec   = to_logmel(noisy_wave).squeeze(0).cpu().numpy()
                spec   = cv2.resize(spec, (IMAGE_SIZE, IMAGE_SIZE))[:, :, np.newaxis]
                fused  = np.concatenate((thermal, spec), axis=2)
                fused_t = test_tf(image=fused)["image"].unsqueeze(0).to(DEVICE)

                logits = model(fused_t)
                preds.append(logits.argmax(1).item())

                feat = model.extract_features(fused_t)
                all_features.append(feat.detach().cpu().numpy().reshape(1, -1))

        cm = confusion_matrix(test_labels, preds)
        plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix — {noise_name.capitalize()} Noise "
                  f"at {snr} dB ({tag.capitalize()} Accuracy)",
            save_path=os.path.join(RESULTS_DIR, f"confusion_{tag}_noisy.png")
        )

        features_array = np.concatenate(all_features, axis=0)
        plot_umap(
            features=features_array,
            labels=np.array(test_labels),
            title=f"UMAP — {noise_name.capitalize()} Noise "
                  f"at {snr} dB ({tag.capitalize()} Accuracy)",
            save_path=os.path.join(RESULTS_DIR, f"umap_{tag}_noisy.png"),
            precomputed=True
        )

    print(f"\nAll evaluation outputs saved to {RESULTS_DIR}/")
