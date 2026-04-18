"""
Fine-tune XLM-RoBERTa for Hinglish sentiment (3-class) using Hugging Face Trainer.

Steps:
  1. Ensure/generate data/hinglish_reviews.csv
  2. Stratified 70/15/15 split
  3. Tokenize + train with early stopping
  4. Evaluate on test set, save plots + confusion matrix + classification report
  5. Save model to model/
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure `src/` is importable when launched as `python -m src.train` from project root
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from data_preparation import clean_review_text, ensure_dataset, stratified_split

# Label mapping (fixed for reproducibility)
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_info() -> str:
    if torch.cuda.is_available():
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    return "CPU"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text and map string labels to integers."""
    out = df.copy()
    out["review_text"] = out["review_text"].astype(str).map(clean_review_text)
    out["labels"] = out["sentiment"].map(LABEL2ID)
    out = out.dropna(subset=["labels"])
    out["labels"] = out["labels"].astype(int)
    return out


def tokenize_batch(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    enc = tokenizer(
        examples["review_text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # Keep labels explicit for batched `.map()` (column merge behavior)
    enc["labels"] = examples["labels"]
    return enc


def compute_metrics_builder():
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_w = f1_score(labels, preds, average="weighted")
        f1_macro = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_macro}

    return compute_metrics


def plot_training_history(log_history: list, out_dir: Path) -> None:
    """Plot loss and accuracy from Trainer state log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    train_loss = []
    eval_loss = []
    eval_acc = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            steps.append(entry.get("step", len(steps)))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_acc.append(entry.get("eval_accuracy", np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if train_loss:
        axes[0].plot(range(len(train_loss)), train_loss, label="Train loss", color="#2563eb")
    if eval_loss:
        axes[0].plot(
            np.linspace(0, len(train_loss) - 1 if train_loss else 1, len(eval_loss)),
            eval_loss,
            label="Validation loss",
            color="#dc2626",
            marker="o",
        )
    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("Logging steps (train) / eval points")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if eval_acc and not all(np.isnan(eval_acc)):
        axes[1].plot(eval_acc, label="Validation accuracy", color="#059669", marker="o")
        axes[1].set_title("Validation accuracy")
        axes[1].set_xlabel("Evaluation epoch index")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No eval accuracy in log", ha="center")

    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=150)
    plt.close()


def plot_confusion_matrix_heatmap(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_and_evaluate(
    csv_path: Path,
    model_out: Path,
    output_dir: Path,
    model_name: str = "xlm-roberta-base",
    max_length: int = 128,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
    early_stopping_patience: int = 3,
) -> Dict[str, Any]:
    set_seed(seed)

    df = ensure_dataset(csv_path, regenerate=False)
    df = preprocess_dataframe(df)

    train_df, val_df, test_df = stratified_split(df, label_col="sentiment", seed=seed)

    print("\n=== Dataset splits (stratified 70/15/15) ===")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Train sentiment counts:\n", train_df["sentiment"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = Dataset.from_pandas(train_df[["review_text", "labels"]])
    val_ds = Dataset.from_pandas(val_df[["review_text", "labels"]])
    test_ds = Dataset.from_pandas(test_df[["review_text", "labels"]])

    def _tok(examples):
        return tokenize_batch(examples, tokenizer, max_length)

    train_ds = train_ds.map(_tok, batched=True, remove_columns=["review_text"])
    val_ds = val_ds.map(_tok, batched=True, remove_columns=["review_text"])
    test_ds = test_ds.map(_tok, batched=True, remove_columns=["review_text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    num_labels = len(LABEL2ID)

    use_fp16 = torch.cuda.is_available()
    # TrainingArguments: `eval_strategy` vs `evaluation_strategy` across transformers versions
    ta_kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir / "checkpoints"),
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        fp16=use_fp16,
        report_to="none",
    )
    sig_ta = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig_ta.parameters:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"
    # Warmup: prefer explicit steps on newer Transformers (warmup_ratio deprecated in v5.2+)
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    total_opt_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = max(1, int(total_opt_steps * 0.1))
    if "warmup_steps" in sig_ta.parameters:
        ta_kwargs["warmup_steps"] = warmup_steps
    elif "warmup_ratio" in sig_ta.parameters:
        ta_kwargs["warmup_ratio"] = 0.1

    training_args = TrainingArguments(**ta_kwargs)

    trainer_kw: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_builder(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    sig_tr = inspect.signature(Trainer.__init__)
    if "tokenizer" in sig_tr.parameters:
        trainer_kw["tokenizer"] = tokenizer
    elif "processing_class" in sig_tr.parameters:
        trainer_kw["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kw)

    print(f"\n=== Training on device: {get_device_info()} ===\n")
    train_result = trainer.train()
    print("\n=== Training complete ===\n")

    # Test evaluation
    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=-1)

    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(num_labels)],
        digits=4,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("\nClassification report:\n")
    print(report)
    print("\nConfusion matrix (rows=true, cols=pred):\n")
    print(cm)

    # Plots
    plot_training_history(trainer.state.log_history, output_dir)
    plot_confusion_matrix_heatmap(
        cm,
        [ID2LABEL[i] for i in range(num_labels)],
        output_dir / "confusion_matrix.png",
    )

    # Save metrics JSON
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(range(num_labels)))
    metrics_payload = {
        "test_accuracy": float(test_acc),
        "per_class": {
            ID2LABEL[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
            for i in range(num_labels)
        },
        "model_name": model_name,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save model + tokenizer + label config for inference
    model_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    label_cfg = {"id2label": {str(k): v for k, v in ID2LABEL.items()}, "label2id": LABEL2ID}
    with open(model_out / "label_config.json", "w", encoding="utf-8") as f:
        json.dump(label_cfg, f, indent=2)

    return {
        "test_accuracy": test_acc,
        "classification_report": report,
        "train_loss": getattr(train_result, "training_loss", None),
        "output_dir": str(output_dir),
        "model_dir": str(model_out),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Train Hinglish sentiment classifier")
    p.add_argument("--data", type=str, default=str(root / "data" / "hinglish_reviews.csv"))
    p.add_argument("--model-out", type=str, default=str(root / "model"))
    p.add_argument("--output-dir", type=str, default=str(root / "outputs"))
    p.add_argument("--model-name", type=str, default="xlm-roberta-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--regenerate-data", action="store_true", help="Regenerate synthetic CSV before training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data)
    if args.regenerate_data:
        from data_preparation import generate_synthetic_dataset, save_csv

        save_csv(generate_synthetic_dataset(seed=args.seed), csv_path)

    ensure_dataset(csv_path, regenerate=False)

    results = train_and_evaluate(
        csv_path=csv_path,
        model_out=Path(args.model_out),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print("\nArtifacts saved:")
    print(f"  Model: {results['model_dir']}")
    print(f"  Plots & metrics: {results['output_dir']}")


if __name__ == "__main__":
    main()
