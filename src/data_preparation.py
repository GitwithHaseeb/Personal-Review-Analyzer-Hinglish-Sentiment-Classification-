"""
Synthetic Lahore-style Roman Hinglish review generator and dataset utilities.

Generates a balanced CSV (1000 samples: ~334 positive, ~333 negative, ~333 neutral),
applies cleaning, and produces stratified train/validation/test splits (70/15/15).
"""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Reproducibility for dataset generation
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Lahore / Pakistan-style Hinglish lexicon (Roman script)
# ---------------------------------------------------------------------------

POS_OPENERS = [
    "Yaar honestly",
    "Bhai dil se bolun",
    "Mast experience tha",
    "Lahore mein itna acha",
    "Seriously impressed",
    "Full paisa vasool",
    "Itna pyara",
    "Zabardast vibe",
    "Love ho gaya",
    "Dil khush ho gaya",
]

POS_MID = [
    "staff bohot cooperative tha",
    "food fresh aur tasty tha",
    "ambience on point thi",
    "service fast thi",
    "portion size bhi theek tha",
    "hygiene top notch thi",
    "family ke sath perfect tha",
    "music aur lighting balanced thi",
    "delivery time bhi reasonable tha",
    "packaging neat thi",
]

POS_CLOSERS = [
    "dobara zaroor aaunga.",
    "friends ko recommend karunga.",
    "full 10/10 vibes.",
    "dil se thank you team.",
    "next weekend phir plan banega.",
    "Lahore mein best lagta hai.",
    "value for money hai bilkul.",
    "maza aa gaya sach mein.",
]

NEG_OPENERS = [
    "Bhai disappointment hi disappointment",
    "Bekar experience raha",
    "Paisa zaya hogaya",
    "Itna ganda kabhi nahi dekha",
    "Seriously pathetic",
    "Time waste pura",
    "Expectations bilkul crush hogayeen",
    "Staff rude tha bilkul",
    "Quality neeche gir gayi",
    "Hype se zyada kuch nahi tha",
]

NEG_MID = [
    "order late aya aur thanda tha",
    "taste flat tha, masala missing",
    "portion chota tha price ke hisaab se",
    "hygiene questionable thi",
    "noise level unbearable tha",
    "billing mein extra charges add kiye",
    "wait time bohot zyada tha",
    "ambience dull aur congested thi",
    "packaging leak hogayi thi",
    "staff argue karne lag gaye",
]

NEG_CLOSERS = [
    "dobara kabhi nahi.",
    "recommend nahi karunga kisi ko.",
    "paise barbad mat karna.",
    "Lahore mein aur options dekh lo.",
    "1/10 bas.",
    "bekar hi bekar.",
    "sorry but sach hai.",
]

NEU_OPENERS = [
    "Theek tha overall",
    "Average sa experience tha",
    "Kuch khaas nahi, kuch bura bhi nahi",
    "Normal restaurant vibes",
    "Mixed feelings hain",
    "Neutral hi bolunga",
    "Itna acha bura dono nahi",
    "Standard expectation ke around tha",
    "Dekh lo agar pasand ho to",
    "Okay okay type tha",
]

NEU_MID = [
    "food average tha, edible tha",
    "service kabhi fast kabhi slow",
    "price thora high lag raha tha",
    "ambience simple thi",
    "taste expected jaisa hi tha",
    "portion size normal tha",
    "location convenient thi",
    "parking thori mushkil thi",
    "menu variety theek thi",
    "noise level manageable tha",
]

NEU_CLOSERS = [
    "try kar sakte ho agar nearby ho.",
    "maybe dobara sochunga.",
    "3.5/5 type feel.",
    "neither wow nor uff.",
    "bas chal jata hai.",
    "expectation medium rakho.",
]


def _pick(rng: random.Random, items: List[str]) -> str:
    return rng.choice(items)


def _maybe_prefix(rng: random.Random, text: str, p: float = 0.35) -> str:
    fillers = [
        "Assalamualaikum, ",
        "Hi bhai, ",
        "So basically ",
        "Quick review: ",
        "Lahore se likh raha hoon — ",
    ]
    if rng.random() < p:
        return rng.choice(fillers) + text
    return text


def _maybe_suffix(rng: random.Random, text: str, p: float = 0.3) -> str:
    tails = [
        " Bas itna hi.",
        " Thanks for reading yaar.",
        " Hope helpful ho.",
        " That's my honest take.",
        " Allah Hafiz.",
    ]
    if rng.random() < p:
        return text + rng.choice(tails)
    return text


def _build_positive_review(rng: random.Random) -> str:
    """Compose a positive Hinglish review with variable length."""
    style = rng.randint(0, 3)
    if style == 0:
        parts = [
            _pick(rng, POS_OPENERS) + ",",
            _pick(rng, POS_MID) + ",",
            _pick(rng, POS_MID) + ".",
            _pick(rng, POS_CLOSERS),
        ]
    elif style == 1:
        parts = [
            _pick(rng, POS_OPENERS) + "!",
            _pick(rng, POS_MID) + ".",
            _pick(rng, POS_CLOSERS),
        ]
    else:
        parts = [
            _pick(rng, POS_OPENERS) + ".",
            _pick(rng, POS_MID) + ",",
            _pick(rng, POS_MID) + ",",
            _pick(rng, POS_MID) + ".",
            _pick(rng, POS_CLOSERS),
        ]
    text = " ".join(parts)
    text = _maybe_prefix(rng, text)
    text = _maybe_suffix(rng, text)
    return clean_review_text(text)


def _build_negative_review(rng: random.Random) -> str:
    style = rng.randint(0, 3)
    if style == 0:
        parts = [
            _pick(rng, NEG_OPENERS) + ",",
            _pick(rng, NEG_MID) + ",",
            _pick(rng, NEG_MID) + ".",
            _pick(rng, NEG_CLOSERS),
        ]
    elif style == 1:
        parts = [
            _pick(rng, NEG_OPENERS) + "!",
            _pick(rng, NEG_MID) + ".",
            _pick(rng, NEG_CLOSERS),
        ]
    else:
        parts = [
            _pick(rng, NEG_OPENERS) + ".",
            _pick(rng, NEG_MID) + ",",
            _pick(rng, NEG_MID) + ",",
            _pick(rng, NEG_MID) + ".",
            _pick(rng, NEG_CLOSERS),
        ]
    text = " ".join(parts)
    text = _maybe_prefix(rng, text)
    text = _maybe_suffix(rng, text)
    return clean_review_text(text)


def _build_neutral_review(rng: random.Random) -> str:
    style = rng.randint(0, 3)
    if style == 0:
        parts = [
            _pick(rng, NEU_OPENERS) + ",",
            _pick(rng, NEU_MID) + ",",
            _pick(rng, NEU_MID) + ".",
            _pick(rng, NEU_CLOSERS),
        ]
    elif style == 1:
        parts = [
            _pick(rng, NEU_OPENERS) + ".",
            _pick(rng, NEU_MID) + ".",
            _pick(rng, NEU_CLOSERS),
        ]
    else:
        parts = [
            _pick(rng, NEU_OPENERS) + "!",
            _pick(rng, NEU_MID) + ",",
            _pick(rng, NEU_MID) + ",",
            _pick(rng, NEU_MID) + ".",
            _pick(rng, NEU_CLOSERS),
        ]
    text = " ".join(parts)
    text = _maybe_prefix(rng, text)
    text = _maybe_suffix(rng, text)
    return clean_review_text(text)


def clean_review_text(text: str) -> str:
    """
    Normalize whitespace, strip junk characters, keep basic Roman + digits + punctuation.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = text.strip()
    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text)
    # Remove excessive repeated punctuation
    text = re.sub(r"([!?.,]){3,}", r"\1\1", text)
    return text.strip()


def generate_synthetic_dataset(
    n_positive: int = 334,
    n_negative: int = 333,
    n_neutral: int = 333,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Build a balanced synthetic Hinglish dataset (default total = 1000 rows).

    Returns columns: review_text, sentiment (string labels).
    """
    rng = random.Random(seed)
    rows: List[Tuple[str, str]] = []

    for _ in range(n_positive):
        rows.append((_build_positive_review(rng), "positive"))
    for _ in range(n_negative):
        rows.append((_build_negative_review(rng), "negative"))
    for _ in range(n_neutral):
        rows.append((_build_neutral_review(rng), "neutral"))

    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=["review_text", "sentiment"])
    df["review_text"] = df["review_text"].map(clean_review_text)
    return df.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "sentiment",
    test_size: float = 0.15,
    val_size_within_train: float = 0.15 / 0.85,
    seed: int = RNG_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    70/15/15 train/val/test split with stratification on `label_col`.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_col],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size_within_train,
        random_state=seed,
        stratify=train_df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def ensure_dataset(
    output_csv: str | Path,
    regenerate: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic data if missing or if regenerate=True.
    """
    output_csv = Path(output_csv)
    if output_csv.exists() and not regenerate:
        return load_csv(output_csv)
    df = generate_synthetic_dataset()
    save_csv(df, output_csv)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hinglish_reviews.csv")
    parser.add_argument(
        "--out",
        type=str,
        default="data/hinglish_reviews.csv",
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"Dataset already exists at {out_path}. Use --force to regenerate.")
        return

    df = generate_synthetic_dataset(seed=args.seed)
    save_csv(df, out_path)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
