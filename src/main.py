# =============================================================================
# main.py
# Full pipeline entry point for Experiment IV:
# Early-Fusion Multimodal CNN for Electromechanical Fault Diagnosis.
#
# Runs the complete pipeline in one command:
#   1. Data loading, pairing, and train/test split        (train.py)
#   2. Model training with AMP and cosine annealing LR   (train.py)
#   3. Training curves saved to results/                  (train.py)
#   4. Clean evaluation on held-out test set              (evaluate.py)
#   5. Noise robustness evaluation (5 types × 6 SNR)     (evaluate.py)
#   6. Confusion matrices and UMAP plots saved            (evaluate.py)
#
# Usage:
#   cd src
#   python main.py
#
# All configuration (paths, hyperparameters) is in config.py.
# Data must be placed under data/ — see data/README.md.
# =============================================================================

from train    import main as run_training
from evaluate import main as run_evaluation


def main():
    print("\n" + "=" * 60)
    print("EXPERIMENT IV — EARLY-FUSION MULTIMODAL CNN")
    print("Thermal + Acoustic Fault Diagnosis Pipeline")
    print("=" * 60)

    # ── Phase 1: Training ──────────────────────────────────────────────────────
    # run_training() returns the test split objects kept in memory
    # so evaluation can run immediately without reloading data from disk.
    print("\n" + "─" * 60)
    print("PHASE 1: Training")
    print("─" * 60)

    test_data = run_training()

    # ── Phase 2: Evaluation ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("PHASE 2: Evaluation")
    print("─" * 60)

    run_evaluation(
        test_thermal  = test_data["test_thermal"],
        test_acoustic = test_data["test_acoustic"],
        test_labels   = test_data["test_labels"],
        test_loader   = test_data["test_loader"],
        test_tf       = test_data["test_tf"],
    )

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("Results saved to results/")
    print("Checkpoint saved to checkpoints/best_fusion.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
