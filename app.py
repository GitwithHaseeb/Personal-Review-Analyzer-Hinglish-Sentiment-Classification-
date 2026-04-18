"""
Gradio web UI for MyHinglishSentiment — Roman Hinglish sentiment analysis.

Run from project root:
    python app.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr

# Project paths
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference import HinglishSentimentPredictor, get_device  # noqa: E402

FEEDBACK_CSV = ROOT / "data" / "user_labeled_feedback.csv"
MODEL_DIR = ROOT / "model"

EMOJI = {"positive": "😊", "negative": "😞", "neutral": "😐"}
THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def load_predictor_safe() -> HinglishSentimentPredictor | None:
    """Load fine-tuned weights if present; otherwise return None."""
    cfg = MODEL_DIR / "config.json"
    if not cfg.exists():
        return None
    try:
        return HinglishSentimentPredictor(MODEL_DIR)
    except Exception:
        return None


_PREDICTOR: HinglishSentimentPredictor | None = load_predictor_safe()


def analyze_text(text: str) -> tuple[str, str, str]:
    """Return formatted prediction, confidence bar label, and raw JSON-ish detail."""
    text = (text or "").strip()
    if not text:
        return "Please enter some Hinglish text.", "", ""

    if _PREDICTOR is None:
        return (
            "### Model not found\n\nTrain first: `python src/train.py` (saves weights to `model/`).",
            "",
            "",
        )

    out = _PREDICTOR.predict(text)
    label = out["label"]
    conf = out["confidence"]
    emoji = EMOJI.get(label, "❔")

    probs = out["probabilities"]
    prob_lines = "\n".join(f"- **{k}**: {v * 100:.2f}%" for k, v in sorted(probs.items()))

    summary = (
        f"## {emoji} **{label.upper()}**\n\n"
        f"**Confidence:** {conf * 100:.2f}%\n\n"
        f"### Class probabilities\n{prob_lines}"
    )
    bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    conf_vis = f"`{bar}` {conf * 100:.1f}%"
    return summary, conf_vis, prob_lines


def append_labeled_sample(review_text: str, sentiment: str) -> str:
    """Append user-labeled row for future retraining (CSV under data/)."""
    review_text = (review_text or "").strip()
    if not review_text:
        return "Enter review text before saving."

    sentiment = (sentiment or "").strip().lower()
    if sentiment not in {"positive", "negative", "neutral"}:
        return "Pick a valid sentiment: positive / negative / neutral."

    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "review_text", "sentiment"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "review_text": review_text,
                "sentiment": sentiment,
            }
        )
    return f"Saved to `{FEEDBACK_CSV.relative_to(ROOT)}`. Merge into training data when you retrain."


def device_badge() -> str:
    dev = get_device()
    return f"**Device:** `{dev}`  ·  **Model dir:** `{MODEL_DIR.name}/`"


with gr.Blocks(theme=THEME, title="MyHinglishSentiment") as demo:
    gr.Markdown(
        """
# 🇵🇰 MyHinglishSentiment
**Lahore-style Roman Hinglish** → **Positive / Negative / Neutral** with confidence scores.

Type food, delivery, or service reviews mixing English + Urdu/Hindi in Roman script.
        """
    )
    status = gr.Markdown(device_badge())

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="Hinglish review",
                placeholder="Yaar biryani mast thi, staff bhi cooperative tha — full recommend!",
                lines=6,
            )
            btn = gr.Button("Analyze sentiment", variant="primary")

        with gr.Column(scale=1):
            out_main = gr.Markdown(label="Prediction")
            out_bar = gr.Markdown(label="Confidence")
            out_detail = gr.Textbox(label="Probabilities (text)", lines=6)

    btn.click(fn=analyze_text, inputs=inp, outputs=[out_main, out_bar, out_detail])

    gr.Markdown("---")
    gr.Markdown("### Contribute labeled data (for future retraining)")
    with gr.Row():
        fb_text = gr.Textbox(label="Review text", lines=3)
        fb_label = gr.Dropdown(
            choices=["positive", "negative", "neutral"],
            label="True sentiment",
            value="positive",
        )
    fb_btn = gr.Button("Save sample to CSV")
    fb_out = gr.Markdown()
    fb_btn.click(fn=append_labeled_sample, inputs=[fb_text, fb_label], outputs=fb_out)

    gr.Markdown(
        """
---
<small>Tip: After collecting feedback, merge `data/user_labeled_feedback.csv` into your main dataset and run `python src/train.py`.</small>
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
