# Early-Fusion Multimodal CNN for Electromechanical Fault Diagnosis

> Combining thermal imaging and acoustic sensing in a single 4-channel CNN  
> to detect machine faults before catastrophic failure — validated across  
> 31 electromechanical machines under real industrial operating conditions.

---

## Business Problem

Unplanned downtime in industrial facilities costs manufacturers an estimated
**$50B annually** (industry estimates vary).
The root cause is reactive maintenance: machines fail without warning,
halting production lines and triggering emergency repair costs.

This project delivers a **data-driven predictive maintenance system** that:
- Detects early-stage machine faults before they escalate
- Classifies fault severity into three actionable states: Healthy, Mild Fault, Heavy Fault
- Operates on low-cost edge hardware (Raspberry Pi 4) with sub-200ms inference latency
- Requires no specialised knowledge from plant operators — outputs a simple health label

---

## Solution Overview

Standard fault detection systems use either thermal imaging **or** acoustic sensing —
not both. Each modality has known failure modes:
- Thermal-only: misses faults that do not generate heat signatures
- Acoustic-only: degrades sharply in noisy industrial environments

This work introduces an **early-fusion multimodal CNN** that concatenates
thermal images and log-mel spectrograms into a single 4-channel input tensor,
allowing a shared CNN backbone to learn cross-modal fault signatures that
neither modality can detect alone.

```
Thermal image (224×224×3)  ──┐
                              ├── [Concatenate] ──► 224×224×4 ──► EarlyFusionNet ──► Fault class
Log-mel spectrogram (224×224×1) ─┘
```

---

## Key Results

| Condition | Accuracy |
|---|---|
| Clean test set (no added noise) | **99.33%** |
| Noise robustness | Evaluated across 5 noise types × 6 SNR levels (−10 to +20 dB) |

**Full results — confusion matrices, UMAP visualisations, noise robustness
tables, and training curves — are withheld pending journal publication.**
They will be made publicly available upon paper acceptance.

> **Reproducibility note:** The 99.33% result was produced with a proprietary
> industrial dataset that cannot be shared. The code, architecture, and pipeline
> are fully reproducible. Exact metrics will differ with alternative datasets.
> See `data/README.md` for guidance on using the CWRU Bearing Dataset as a
> public proxy.

---

## Architecture

`EarlyFusionNet` is a lightweight CNN built on **depthwise separable convolutions**
(MobileNet-style), designed for deployment on resource-constrained edge devices.

```
Input [B, 4, 224, 224]
  │
  ▼
Stem: Conv2d(4→32, 3×3, stride=2) + BN + ReLU          → [B,  32, 112, 112]
  │
  ▼
Block 1: DS-Conv(32→64,  stride=2) × 2                  → [B,  64,  56,  56]
Block 2: DS-Conv(64→128, stride=2) × 3                  → [B, 128,  28,  28]
Block 3: DS-Conv(128→256,stride=2) × 4                  → [B, 256,  14,  14]
Block 4: DS-Conv(256→512,stride=2) × 3                  → [B, 512,   7,   7]
  │
  ▼
Conv head: Conv2d(512→1024, 3×3, stride=2) + BN + ReLU → [B, 1024,  4,   4]
  │
  ▼
GlobalAvgPool → Flatten → Dropout(0.3) → Linear(1024→3)
  │
  ▼
Output: [B, 3]  (Healthy / Mild_Fault / Heavy_Fault)
```

**Depthwise separable convolution** replaces each standard 3×3 conv with:
1. A depthwise conv (spatial filtering, one filter per channel)
2. A pointwise 1×1 conv (channel mixing)

This reduces FLOPs by ~8–9× while maintaining representational capacity —
critical for real-time inference on Raspberry Pi 4.

---

## IoT Deployment Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Edge Device (Raspberry Pi 4)              │
│                                                              │
│  INMP441 MEMS mic ──► acoustic segment ──► log-mel spec     │
│  Fluke Ti400 PRO  ──► thermal image                         │
│                           │                                  │
│                    [Early Fusion CNN]                        │
│                    ONNX Runtime inference                    │
│                    Latency: < 200ms                          │
└──────────────────────────┬──────────────────────────────────┘
                           │ MQTT (Mosquitto broker)
                           ▼
              Node-RED dashboard ──► real-time fault alert
```

See `src/model.py` for ONNX export instructions (commented block at end of file).

---

## Repository Structure

```
├── src/
│   ├── config.py       ← all hyperparameters and paths
│   ├── dataset.py      ← data loading, pairing, normalisation, Dataset class
│   ├── model.py        ← EarlyFusionNet + ONNX export (commented)
│   ├── train.py        ← training loop entry point
│   ├── evaluate.py     ← clean + noise robustness evaluation
│   └── visualize.py    ← confusion matrix, training curves, UMAP plots
│
├── data/
│   ├── thermal/        ← thermal images (not included — see data/README.md)
│   ├── audio/          ← acoustic recordings (not included)
│   └── noise_sources/  ← noise .mp3 files (not included — see noise_sources/README.md)
│
├── results/            ← confusion matrices, training curves, UMAP plots
├── checkpoints/        ← saved model weights (best_fusion.pth)
├── requirements.txt
└── README.md
```

---

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/predictive-maintenance-multimodal-cnn.git
cd predictive-maintenance-multimodal-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare data
Follow the structure described in `data/README.md`.
For a public proxy dataset, see the CWRU Bearing Dataset instructions there.

### 4. Configure paths and hyperparameters
Edit `src/config.py` — all settings are in one place.

### 5. Train
```bash
cd src
python train.py
```
Checkpoints saved to `checkpoints/best_fusion.pth`.
Training curves saved to `results/`.

### 6. Evaluate
```python
# In a notebook or script, after training:
from evaluate import main as run_eval
run_eval(test_thermal, test_acoustic, test_labels, test_loader, test_tf)
```

---

## Training Details

| Parameter | Value |
|---|---|
| Epochs | 140 |
| Batch size | 16 |
| Optimiser | AdamW (lr=5e-4, weight decay=1e-4) |
| LR schedule | Cosine annealing with warm restarts (T₀=10) |
| Loss | Cross-entropy with class weights (balanced) |
| Mixed precision | AMP (automatic, CUDA only) |
| Train / test split | 80 / 20 stratified |

**Data augmentation (training only):**
horizontal flip, rotation (±15°), brightness/contrast jitter,
Gaussian noise injection, coarse dropout, grid distortion.

**Normalisation:**
- Thermal channels (RGB): ImageNet mean/std
- Acoustic channel (log-mel): zero mean / unit variance computed from training split

---

## Experimental Context

This repository implements **Experiment** from doctoral thesis research on
IoT-integrated predictive maintenance for electromechanical machines.

---

## Limitations

- Dataset is proprietary and not publicly available. Results are not
  directly reproducible without equivalent industrial data collection.
- Train and test sets are disjoint; however, the same test set is used
  for both validation (checkpoint selection) and final evaluation.
  This is optimistic — a separate validation set would give a stricter bound.
- Noise robustness uses artificially mixed noise rather than recordings
  made in genuinely noisy environments, which may overestimate real-world robustness.

---

## License and Rights

© [Verasis Kour], [2025]. All rights reserved.

This repository is shared for **portfolio and review purposes only**.
No part of this code, methodology, or results may be reproduced,
distributed, adapted, or used in any form without explicit written
permission from the author.

This work is associated with an unpublished doctoral thesis currently
under examination. Commercial use is strictly prohibited.

---

## Citation

If you reference this work, please cite the associated paper:

```
[to be updated]
```
 
