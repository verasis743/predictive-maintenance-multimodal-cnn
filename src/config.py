# =============================================================================
# config.py
# Central configuration for Experiment 4:
# Early-Fusion Multimodal CNN for Electromechanical Fault Diagnosis
#
# All hyperparameters, paths, and constants are defined here.
# Change values in this file only — do not hardcode values elsewhere.
# =============================================================================

import os

# ── Device ────────────────────────────────────────────────────────────────────
# Training uses GPU if available, falls back to CPU automatically.
# Mixed-precision (AMP) is enabled when CUDA is available.
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Class definitions ─────────────────────────────────────────────────────────
# Three health states monitored across electromechanical machines.
CLASS_NAMES   = ["Healthy", "Mild_Fault", "Heavy_Fault"]
NUM_CLASSES   = len(CLASS_NAMES)
CLASS_TO_IDX  = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── Audio configuration ───────────────────────────────────────────────────────
SAMPLE_RATE     = 16000   # Hz — target sample rate for all audio
SEGMENT_LEN     = 5.0     # seconds — length of each acoustic segment
DISCARD_START   = 10.0    # seconds — startup transients discarded from recording head
N_FFT           = 1024    # FFT window size for log-mel spectrogram
HOP_LENGTH      = 512     # FFT hop (50% overlap)
N_MELS          = 128     # number of mel filterbank bins
F_MIN           = 20      # Hz — low-frequency cutoff
F_MAX           = 8000    # Hz — high-frequency cutoff (Nyquist for 16 kHz)

# ── Image / fusion configuration ─────────────────────────────────────────────
# Both thermal images and log-mel spectrograms are resized to IMAGE_SIZE x IMAGE_SIZE
# before channel-wise concatenation into a 4-channel fused input tensor.
IMAGE_SIZE      = 224     # pixels (H = W)
IN_CHANNELS     = 4       # 3 thermal (RGB) + 1 acoustic (log-mel)

# ── Training configuration ────────────────────────────────────────────────────
MAX_EPOCHS      = 140
BATCH_SIZE      = 16
LEARNING_RATE   = 5e-4
WEIGHT_DECAY    = 1e-4
DROPOUT_RATE    = 0.3
RANDOM_STATE    = 42      # seed for reproducible train/test split
TEST_SIZE       = 0.2     # fraction of data held out for evaluation

# ── Noise robustness evaluation ───────────────────────────────────────────────
# Model is evaluated under 5 real-world noise types across 6 SNR levels.
# Each condition is repeated 3 times; mean ± std is reported.
SNR_LEVELS      = [20, 10, 5, 0, -5, -10]   # dB
NOISE_TYPES     = ["white", "factory", "cafeteria", "traffic", "rain"]
EVAL_RUNS       = 3       # repetitions per noise × SNR condition

# ── Paths (all relative to repo root) ────────────────────────────────────────
# Do NOT use absolute paths. Keep everything relative so the repo runs
# on any machine after cloning.
THERMAL_DIR     = os.path.join("data", "thermal")
AUDIO_DIR       = os.path.join("data", "audio")
NOISE_DIR       = os.path.join("data", "noise_sources")
CHECKPOINT_DIR  = "checkpoints"
RESULTS_DIR     = "results"

# Subdirectory layout expected under THERMAL_DIR and AUDIO_DIR:
#   data/thermal/Healthy/       ← thermal images (.jpg / .png)
#   data/thermal/Mild_Fault/
#   data/thermal/Heavy_Fault/
#   data/audio/Healthy/         ← acoustic recordings (.wav)
#   data/audio/Mild_Fault/
#   data/audio/Heavy_Fault/
#   data/noise_sources/         ← factory.mp3, cafeteria.mp3, traffic.mp3, rain.mp3
#                                 (white noise is generated programmatically)

# ── ImageNet normalisation stats for thermal channels (RGB) ──────────────────
# Standard ImageNet mean/std used for the 3 thermal channels.
# The 4th channel (log-mel spectrogram) is normalised to zero mean / unit
# variance computed from the training split (see dataset.py).
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
