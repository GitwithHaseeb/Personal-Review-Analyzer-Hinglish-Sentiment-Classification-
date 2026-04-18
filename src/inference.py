"""
Load a fine-tuned model for Hinglish sentiment inference with confidence scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Default label order must match training
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HinglishSentimentPredictor:
    """Thin wrapper around HF tokenizer + sequence classification model."""

    def __init__(self, model_dir: str | Path, device: torch.device | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Prefer saved label maps if present
        config_path = self.model_dir / "label_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
            self.label2id = cfg["label2id"]
        else:
            self.id2label = ID2LABEL
            self.label2id = LABEL2ID

    @torch.inference_mode()
    def predict(self, text: str, max_length: int = 128) -> Dict[str, Any]:
        """Return predicted label, confidence, and full probability vector."""
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        label = self.id2label[pred_id]
        confidence = float(probs[pred_id])
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {self.id2label[i]: float(probs[i]) for i in range(len(probs))},
        }

    def predict_batch(self, texts: List[str], max_length: int = 128) -> List[Dict[str, Any]]:
        return [self.predict(t, max_length=max_length) for t in texts]


def load_predictor(model_dir: str | Path | None = None) -> HinglishSentimentPredictor:
    """Load predictor from `model/` relative to project root by default."""
    root = Path(__file__).resolve().parent.parent
    path = Path(model_dir) if model_dir else root / "model"
    return HinglishSentimentPredictor(path)
