# =============================================================================
# dataset.py
# Data loading, acoustic segmentation, thermal-acoustic pairing,
# spectrogram normalisation, and PyTorch Dataset class.
#
# Key design decisions:
# - Thermal images and acoustic recordings are paired by health class.
#   If the number of thermal images and acoustic segments differ per class,
#   thermal images are tiled (repeated cyclically) to match the acoustic count.
# - Log-mel spectrograms are normalised to zero mean / unit variance using
#   statistics computed exclusively from the training split (no data leakage).
# - Both thermal images and spectrograms are resized to IMAGE_SIZE x IMAGE_SIZE
#   before concatenation into a 4-channel fused tensor.
# =============================================================================

import os
import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    CLASS_NAMES, CLASS_TO_IDX,
    SAMPLE_RATE, SEGMENT_LEN, DISCARD_START,
    N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    THERMAL_DIR, AUDIO_DIR
)


# ── Log-mel spectrogram transform ─────────────────────────────────────────────
# Converts a raw waveform tensor [1, T] to a log-mel spectrogram [128, time_frames].
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
)


def to_logmel(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert a mono waveform tensor to a log-mel spectrogram.

    Args:
        waveform: Tensor of shape [1, T] at SAMPLE_RATE Hz.

    Returns:
        Log-mel spectrogram of shape [N_MELS, time_frames].
        A small epsilon (1e-10) is added before log to avoid log(0).
    """
    spec = _mel_transform(waveform)
    log_spec = torch.log(spec + 1e-10)
    return log_spec.squeeze(0)  # [N_MELS, time_frames]


# ── Thermal image loader ───────────────────────────────────────────────────────

def load_thermal_paths(root: str) -> tuple:
    """
    Scan the thermal image directory and return file paths with labels.

    Expected folder structure:
        root/Healthy/        ← .jpg / .jpeg / .png / .bmp
        root/Mild_Fault/
        root/Heavy_Fault/

    Args:
        root: Path to the thermal image root directory.

    Returns:
        Tuple of (paths, labels) where labels are integer class indices.
    """
    paths, labels = [], []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for cls in CLASS_NAMES:
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Thermal class folder not found: {folder}\n"
                f"Expected structure: {root}/Healthy/, {root}/Mild_Fault/, {root}/Heavy_Fault/"
            )
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                paths.append(os.path.join(folder, f))
                labels.append(CLASS_TO_IDX[cls])
    return paths, labels


# ── Acoustic loader and segmenter ─────────────────────────────────────────────

def load_and_segment(path: str) -> list:
    """
    Load an audio file, resample to SAMPLE_RATE, discard startup transients,
    and split into fixed-length non-overlapping segments.

    Args:
        path: Path to a .wav or .mp3 audio file.

    Returns:
        List of waveform tensors, each of shape [1, SEGMENT_LEN * SAMPLE_RATE].
        Incomplete final segments are discarded.
    """
    waveform, orig_sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target sample rate
    if orig_sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)(waveform)

    # Discard initial seconds to remove startup transients
    discard_samples = int(DISCARD_START * SAMPLE_RATE)
    waveform = waveform[:, discard_samples:]

    # Segment into fixed-length clips
    samples_per_seg = int(SEGMENT_LEN * SAMPLE_RATE)
    segments = []
    for start in range(0, waveform.shape[1], samples_per_seg):
        end = start + samples_per_seg
        if end > waveform.shape[1]:
            break  # discard incomplete final segment
        segments.append(waveform[:, start:end])

    return segments


def load_acoustic_segments(root: str) -> tuple:
    """
    Load and segment all acoustic recordings under the audio root directory.

    Expected folder structure:
        root/Healthy/        ← .wav / .mp3
        root/Mild_Fault/
        root/Heavy_Fault/

    Args:
        root: Path to the audio root directory.

    Returns:
        Tuple of (waveforms, labels) where waveforms is a list of tensors
        [1, SEGMENT_LEN * SAMPLE_RATE] and labels are integer class indices.
    """
    waveforms, labels = [], []
    valid_exts = {".wav", ".mp3"}
    for cls in CLASS_NAMES:
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Audio class folder not found: {folder}"
            )
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                path = os.path.join(folder, f)
                segs = load_and_segment(path)
                waveforms.extend(segs)
                labels.extend([CLASS_TO_IDX[cls]] * len(segs))
    return waveforms, labels


# ── Thermal-acoustic pairing ───────────────────────────────────────────────────

def pair_thermal_acoustic(
    thermal_paths: list, thermal_labels: list,
    acoustic_waves: list, acoustic_labels: list
) -> tuple:
    """
    Pair thermal images with acoustic segments within each health class.

    Since thermal images and acoustic segments may differ in count per class,
    thermal images are tiled (repeated cyclically) to match the acoustic
    segment count. This ensures every acoustic segment has a corresponding
    thermal image from the same health state.

    Args:
        thermal_paths:   List of thermal image file paths.
        thermal_labels:  Integer class labels for thermal images.
        acoustic_waves:  List of waveform tensors.
        acoustic_labels: Integer class labels for acoustic segments.

    Returns:
        Tuple of (paired_thermal, paired_acoustic, paired_labels).
    """
    paired_thermal, paired_acoustic, paired_labels = [], [], []

    for cls_idx in range(len(CLASS_NAMES)):
        cls_thermal  = [p for p, l in zip(thermal_paths,  thermal_labels)  if l == cls_idx]
        cls_acoustic = [w for w, l in zip(acoustic_waves, acoustic_labels) if l == cls_idx]

        n_thermal  = len(cls_thermal)
        n_acoustic = len(cls_acoustic)

        if n_thermal == 0 or n_acoustic == 0:
            raise ValueError(
                f"Class '{CLASS_NAMES[cls_idx]}' has {n_thermal} thermal images "
                f"and {n_acoustic} acoustic segments. Both must be > 0."
            )

        # Tile thermal images to match acoustic segment count
        thermal_tiled = np.tile(cls_thermal, (n_acoustic // n_thermal + 1))[:n_acoustic]

        paired_thermal.extend(thermal_tiled)
        paired_acoustic.extend(cls_acoustic)
        paired_labels.extend([cls_idx] * n_acoustic)

    return paired_thermal, paired_acoustic, paired_labels


# ── Spectrogram normalisation ──────────────────────────────────────────────────

def compute_spectrogram_stats(train_waveforms: list) -> tuple:
    """
    Compute mean and standard deviation of log-mel spectrograms
    from the training split only (prevents data leakage from test set).

    Args:
        train_waveforms: List of waveform tensors from the training split.

    Returns:
        Tuple of (mean, std) as scalar floats.
    """
    train_specs = [to_logmel(w).flatten() for w in train_waveforms]
    flat = torch.cat(train_specs)
    return flat.mean().item(), flat.std().item()


# ── Albumentations transforms ─────────────────────────────────────────────────

def build_transforms(spec_mean: float, spec_std: float) -> tuple:
    """
    Build Albumentations augmentation pipelines for training and evaluation.

    The 4-channel input [thermal_R, thermal_G, thermal_B, log_mel] is
    normalised channel-wise:
      - Channels 0-2 (thermal): ImageNet mean/std
      - Channel 3  (acoustic):  zero mean / unit variance from training data

    Args:
        spec_mean: Mean of log-mel spectrograms computed from training split.
        spec_std:  Std  of log-mel spectrograms computed from training split.

    Returns:
        Tuple of (train_transform, test_transform).
    """
    mean_4ch = IMAGENET_MEAN + [spec_mean]
    std_4ch  = IMAGENET_STD  + [spec_std]

    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.GridDistortion(p=0.3),
        A.Normalize(mean=mean_4ch, std=std_4ch),
        ToTensorV2(),
    ])

    test_tf = A.Compose([
        A.Normalize(mean=mean_4ch, std=std_4ch),
        ToTensorV2(),
    ])

    return train_tf, test_tf


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for paired thermal + acoustic fault classification.

    Each sample is constructed by:
      1. Loading a thermal image and resizing to IMAGE_SIZE x IMAGE_SIZE (RGB).
      2. Converting the paired acoustic waveform to a log-mel spectrogram,
         resizing to IMAGE_SIZE x IMAGE_SIZE, and adding a channel dimension.
      3. Concatenating along the channel axis to form a 4-channel tensor
         [IMAGE_SIZE, IMAGE_SIZE, 4] before applying the transform.

    Args:
        thermal_paths:  List of thermal image file paths.
        acoustic_waves: List of waveform tensors [1, T].
        labels:         Integer class labels.
        transform:      Albumentations transform pipeline (train or test).
    """

    def __init__(
        self,
        thermal_paths:  list,
        acoustic_waves: list,
        labels:         list,
        transform=None
    ):
        self.thermal_paths  = thermal_paths
        self.acoustic_waves = acoustic_waves
        self.labels         = labels
        self.transform      = transform

    def __len__(self) -> int:
        return len(self.thermal_paths)

    def __getitem__(self, i: int) -> tuple:
        # ── Load and resize thermal image ──────────────────────────────────
        thermal = cv2.imread(self.thermal_paths[i])
        if thermal is None:
            raise ValueError(f"Failed to load thermal image: {self.thermal_paths[i]}")
        thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
        thermal = cv2.resize(thermal, (IMAGE_SIZE, IMAGE_SIZE))  # [H, W, 3]

        # ── Convert acoustic waveform to log-mel spectrogram ───────────────
        # to_logmel returns [N_MELS, time_frames]; we resize to IMAGE_SIZE x IMAGE_SIZE
        # and add a channel dimension to match the thermal spatial resolution.
        spec = to_logmel(self.acoustic_waves[i])          # [N_MELS, time_frames]
        spec = spec.squeeze(0).cpu().numpy()               # [N_MELS, time_frames]
        spec = cv2.resize(spec, (IMAGE_SIZE, IMAGE_SIZE))  # [IMAGE_SIZE, IMAGE_SIZE]
        spec = spec[:, :, np.newaxis]                      # [IMAGE_SIZE, IMAGE_SIZE, 1]

        # ── Early fusion: concatenate along channel axis ───────────────────
        # Result: [IMAGE_SIZE, IMAGE_SIZE, 4]  (3 thermal + 1 acoustic)
        fused = np.concatenate((thermal, spec), axis=2)

        # ── Apply transform (normalisation + optional augmentation) ────────
        if self.transform:
            fused = self.transform(image=fused)["image"]  # [4, IMAGE_SIZE, IMAGE_SIZE]

        return fused, self.labels[i]
