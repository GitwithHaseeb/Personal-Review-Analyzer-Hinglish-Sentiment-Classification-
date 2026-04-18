"""
Create demo outputs/ files so build_word_report.py can embed figures without a full GPU training run.

Run: python scripts/generate_report_placeholder_outputs.py
Then: python scripts/build_word_report.py

These are illustrative only — replace by running `python src/train.py` for real experiment results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MARKER = OUT / ".demo_placeholder_for_report"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(
        "Demo metrics/plots for Word report only. Delete this file and outputs/*.png after real training.\n",
        encoding="utf-8",
    )

    # Plausible demo metrics (not from a real training run)
    metrics = {
        "test_accuracy": 0.9453,
        "model_name": "xlm-roberta-base",
        "per_class": {
            "negative": {"precision": 0.9421, "recall": 0.9388, "f1": 0.9404},
            "neutral": {"precision": 0.9312, "recall": 0.9488, "f1": 0.9399},
            "positive": {"precision": 0.9610, "recall": 0.9467, "f1": 0.9538},
        },
        "_note": "DEMO placeholder — run python src/train.py for real metrics",
    }
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Fake training curves
    epochs = np.arange(1, 9)
    train_loss = 2.1 * np.exp(-0.45 * epochs) + 0.08 + np.random.RandomState(42).randn(8) * 0.02
    val_loss = 2.0 * np.exp(-0.4 * epochs) + 0.12 + np.random.RandomState(43).randn(8) * 0.03
    val_acc = 0.55 + 0.4 * (1 - np.exp(-0.5 * epochs)) + np.random.RandomState(44).randn(8) * 0.01
    val_acc = np.clip(val_acc, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, "b-o", label="Train loss")
    axes[0].plot(epochs, val_loss, "r-s", label="Val loss")
    axes[0].set_title("Loss (demo illustration)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, val_acc, "g-o", label="Val accuracy")
    axes[1].set_ylim(0.4, 1.05)
    axes[1].set_title("Validation accuracy (demo illustration)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("DEMO — replace with real training_history.png after training", fontsize=10, color="gray")
    plt.tight_layout()
    plt.savefig(OUT / "training_history.png", dpi=150)
    plt.close()

    # Fake confusion matrix (50 samples per class typical split noise)
    cm = np.array([[47, 2, 1], [3, 46, 1], [1, 2, 47]])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"],
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion matrix (demo illustration)")
    plt.figtext(0.5, 0.01, "DEMO — replace with real confusion_matrix.png after training", ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Wrote demo outputs to {OUT}")
    print("Next: python scripts/build_word_report.py")


if __name__ == "__main__":
    main()
