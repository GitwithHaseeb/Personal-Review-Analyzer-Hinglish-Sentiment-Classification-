"""
Build notebooks/MyHinglishSentiment.ipynb for Google Colab (single all-in-one notebook).

Run from project root:
    python scripts/build_colab_allinone_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "MyHinglishSentiment.ipynb"

COLAB_ROOT = "/content/MyHinglishSentiment"


def cell_md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def cell_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    files = [
        ("__init__.py", (SRC / "__init__.py").read_text(encoding="utf-8")),
        ("data_preparation.py", (SRC / "data_preparation.py").read_text(encoding="utf-8")),
        ("train.py", (SRC / "train.py").read_text(encoding="utf-8")),
        ("inference.py", (SRC / "inference.py").read_text(encoding="utf-8")),
    ]

    cells = []

    cells.append(
        cell_md(
            """# MyHinglishSentiment

Single-notebook Colab workflow: installs dependencies, writes `src/*.py`, trains, shows plots, optional Gradio.

**Runtime → Change runtime type → GPU** recommended. Run cells top to bottom (or Run all).
"""
        )
    )

    cells.append(
        cell_code(
            """%pip install -q torch transformers datasets accelerate pandas scikit-learn matplotlib seaborn sentencepiece safetensors gradio tqdm ipython

import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""
        )
    )

    cells.append(
        cell_code(
            f"""import os
from pathlib import Path

ROOT = Path("{COLAB_ROOT}")
for sub in ("src", "data", "model", "outputs", "reports"):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
print(ROOT.resolve())
"""
        )
    )

    for fname, body in files:
        path = f"{COLAB_ROOT}/src/{fname}"
        src = f"%%writefile {path}\n" + body
        if not src.endswith("\n"):
            src += "\n"
        cells.append(cell_code(src))

    cells.append(
        cell_code(
            f"""%cd {COLAB_ROOT}
!python src/train.py --epochs 10 --batch-size 16 --lr 2e-5
"""
        )
    )

    cells.append(
        cell_code(
            f"""from IPython.display import Image, display
from pathlib import Path
import json

out = Path("{COLAB_ROOT}") / "outputs"
for name in ("training_history.png", "confusion_matrix.png"):
    p = out / name
    if p.exists():
        display(Image(filename=str(p)))
m = out / "metrics.json"
print(json.dumps(json.loads(m.read_text(encoding="utf-8")), indent=2) if m.exists() else "No metrics.json yet.")
"""
        )
    )

    # Use chr(10) so the .ipynb source never breaks on embedded newlines in "\\n" strings
    gradio_src = """import os, sys
os.chdir("__ROOT__")
sys.path.insert(0, "__ROOT__/src")
import gradio as gr
from inference import HinglishSentimentPredictor

pred = HinglishSentimentPredictor("__ROOT__/model")
EMO = {"positive": "😊", "negative": "😞", "neutral": "😐"}

def go(text):
    text = (text or "").strip()
    if not text:
        return "Enter text."
    o = pred.predict(text)
    e = EMO.get(o["label"], "")
    nl = chr(10)
    lines = [f"{k}: {v * 100:.1f}%" for k, v in o["probabilities"].items()]
    probs = nl.join(lines)
    head = f"{e} **{o['label'].upper()}** — confidence {o['confidence'] * 100:.1f}%"
    return head + nl + nl + probs

demo = gr.Interface(fn=go, inputs=gr.Textbox(lines=4, label="Hinglish review"), outputs=gr.Markdown())
demo.launch(share=True)
"""
    cells.append(cell_code(gradio_src.replace("__ROOT__", COLAB_ROOT)))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": []},
        },
        "cells": cells,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
