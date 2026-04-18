# MyHinglishSentiment

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-XLM--R-yellow)](https://huggingface.co/xlm-roberta-base)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio)](https://gradio.app/)

**Roman-script Hinglish sentiment classification** (English mixed with Urdu/Hindi vocabulary, *Lahore / Pakistan–style informal text*) into **positive**, **negative**, or **neutral** using a fine-tuned **[XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)** model and the Hugging Face **Trainer** API.

| | |
|---|---|
| **Task** | 3-class sentiment |
| **Data** | 1,000 synthetic balanced reviews → `data/hinglish_reviews.csv` |
| **Split** | Stratified 70% / 15% / 15% (train / validation / test) |
| **Expected test acc.** | ~**92–96%** on the synthetic benchmark (varies by seed/hardware) |

> **Note:** Synthetic data is for prototyping. Real-world accuracy depends on your domain; collect labels via `data/user_labeled_feedback.csv` (from the Gradio UI) and retrain.

---

## Table of contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Publish to GitHub](#publish-to-github)
- [Clone & install](#clone--install)
- [Train & run the app](#train--run-the-app)
- [Google Colab](#google-colab)
- [Word report](#word-report-docx)
- [Dataset & model](#dataset--model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Synthetic Lahore-style Hinglish** dataset (balanced classes), generated in `src/data_preparation.py`
- **Fine-tuning** with early stopping, metrics, **confusion matrix**, and **training curves** under `outputs/` (ignored by git — generate locally)
- **Gradio** web UI (`app.py`): prediction, confidence, emoji, optional CSV export for future retraining
- **Single Colab notebook** (`notebooks/MyHinglishSentiment.ipynb`) to run the full pipeline without uploading the whole repo as a zip
- **Optional Word report** via `scripts/build_word_report.py`

---

## Tech stack

| Component | Choice |
|-----------|--------|
| ML | [PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets) |
| Model | `xlm-roberta-base` + sequence classification head |
| UI | [Gradio](https://gradio.app/) |
| Metrics / plots | scikit-learn, matplotlib, seaborn |

---

## Repository structure

```text
MyHinglishSentiment/
├── app.py                 # Gradio UI
├── train.py               # Entry point → src.train
├── requirements.txt
├── data/                  # hinglish_reviews.csv (created on first train if missing)
├── model/                 # Fine-tuned weights (created after training)
├── outputs/               # Plots, metrics (gitignored — not in remote by default)
├── notebooks/
│   └── MyHinglishSentiment.ipynb
├── reports/               # Generated .docx report (optional)
├── scripts/
│   ├── build_word_report.py
│   └── build_colab_allinone_notebook.py
└── src/
    ├── data_preparation.py
    ├── train.py
    └── inference.py
```

After cloning, **`outputs/`** and large checkpoints are **not** tracked (see `.gitignore`). Run training to recreate them.

---

## Publish to GitHub

On [GitHub](https://github.com/new), create a new repository named **`MyHinglishSentiment`** (no README if you already have one locally). Then:

```bash
cd MyHinglishSentiment
git init
git add .
git commit -m "Initial commit: MyHinglishSentiment"
git branch -M main
git remote add origin https://github.com/GitwithHaseeb/MyHinglishSentiment.git
git push -u origin main
```

If you **fork** or **rename** the repository, update clone URLs and the Colab link below to match your account and default branch (`main` vs `master`).

---

## Clone & install

```bash
git clone https://github.com/GitwithHaseeb/MyHinglishSentiment.git
cd MyHinglishSentiment
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Use **Python 3.10–3.12** for best compatibility with PyTorch and Transformers. For **NVIDIA GPUs**, install the matching CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Train & run the app

From the repository root:

```bash
python src/train.py
# or: python train.py
```

This creates `data/hinglish_reviews.csv` if needed, saves the model under **`model/`**, and writes plots + `metrics.json` to **`outputs/`**.

Launch the web UI:

```bash
python app.py
```

Default URL: **http://127.0.0.1:7860**

---

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GitwithHaseeb/MyHinglishSentiment/blob/main/notebooks/MyHinglishSentiment.ipynb)

Works once the repo is **public** on GitHub at the URL above. If the badge fails (private repo or not pushed yet), upload `notebooks/MyHinglishSentiment.ipynb` to Colab manually.

**Steps:** Runtime → **Change runtime type** → **GPU** → run all cells. The notebook writes `src/*.py` under `/content/MyHinglishSentiment` and runs training.

Regenerate the notebook from source after editing `src/`:

```bash
python scripts/build_colab_allinone_notebook.py
```

**Alternative (full repo on Colab):** upload the zipped project or clone your repo, then:

```python
%cd MyHinglishSentiment
!pip install -r requirements.txt
!python src/train.py
```

---

## Word report (`.docx`)

Generates **`reports/MyHinglishSentiment_Project_Report.docx`**: **Abstract**, **keywords**, **Colab notebook** section, methodology, hyperparameter table, embedded **PNG** figures from `outputs/`, metrics tables, and appendices (`metrics.json`, `label_config.json`).

1. **Preferred:** train first so `outputs/` has real plots: `python src/train.py`, then `python scripts/build_word_report.py`.
2. **If training is not possible locally** (e.g. low RAM), generate illustrative figures for the report only:  
   `python scripts/generate_report_placeholder_outputs.py` then `python scripts/build_word_report.py`  
   (the Word file will state that placeholders were used; replace with real training when you can).

3. Open **`reports/MyHinglishSentiment_Project_Report.docx`** in Word or LibreOffice.

---

## Dataset & model

| Column | Description |
|--------|-------------|
| `review_text` | Roman Hinglish review text |
| `sentiment` | `positive` · `negative` · `neutral` |

- **Architecture:** XLM-RoBERTa encoder + 3-way classifier head; **max length** 128 tokens; cross-entropy loss.
- **Optional feedback:** `data/user_labeled_feedback.csv` (from the UI) can be merged into future training pipelines.

---

## Results

On the **synthetic** benchmark, **test accuracy** is typically in the **92–96%** range. See `outputs/metrics.json` and the classification report printed after training.

**Screenshots (optional for this README):** add `docs/ui.png` / `docs/training.png` and link them here after you run the project.

---

## Contributing

1. Fork the repository  
2. Create a branch: `git checkout -b feature/your-feature`  
3. Commit changes: `git commit -m "Add: short description"`  
4. Push and open a **Pull Request**

Bug reports and ideas are welcome via **Issues**.

---

## License

This project is provided **as-is** for learning and research. Third-party libraries (PyTorch, Hugging Face Transformers, Gradio, etc.) remain under their respective licenses. Add a `LICENSE` file (e.g. MIT) in your repo if you want a standard open-source terms.

---

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)  
- [Gradio](https://gradio.app/)  
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) pretrained weights  

---

## Citation

If you use this project in academic work, cite the underlying **XLM-R** paper and this repository:

```bibtex
@misc{myhinglishsentiment2026,
  title        = {MyHinglishSentiment: Roman Hinglish Sentiment Classification},
  howpublished = {\url{https://github.com/GitwithHaseeb/MyHinglishSentiment}},
  year         = {2026}
}
```
