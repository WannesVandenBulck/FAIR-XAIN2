#!/usr/bin/env python
"""
Sentiment analysis script.

For each narrative, runs two sentiment analysers:
  VADER  (rule-based, lexicon-driven)
    - vader_neg / vader_neu / vader_pos:  proportion scores in [0, 1]
    - vader_compound:                     normalised score in [-1, 1]

  SiEBERT  (siebert/sentiment-roberta-large-english, Hartmann et al. 2023)
    - siebert_label:    POSITIVE or NEGATIVE
    - siebert_score:    model confidence in [0, 1]
    - siebert_compound: signed confidence — score if POSITIVE else -score → [-1, 1]

One row is written per narrative instance so that downstream fairness analysis
can compare scores across protected-attribute groups.
Results are saved to results/fairness_eval/sentiment.csv.

Dependencies (beyond base requirements):
  pip install vaderSentiment transformers torch
"""

import json
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

ROOT = Path(__file__).parent.parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
DATASETS_TO_EVAL = ["saudi"]    # or ["credit", "law", "saudi", "student"]
CONDITIONS_TO_EVAL = None       # None = all conditions found on disk
# SiEBERT truncates inputs silently to 512 tokens (RoBERTa limit)
SIEBERT_MODEL = "siebert/sentiment-roberta-large-english"
# ============================================================

NARRATIVES_DIR = ROOT / "results" / "narratives"
OUTPUT_FILE = ROOT / "results" / "fairness_eval" / "sentiment.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyse(text, vader, siebert):
    vs = vader.polarity_scores(text)
    sb = siebert(text, truncation=True, max_length=512)[0]
    compound = sb["score"] if sb["label"] == "POSITIVE" else -sb["score"]
    return {
        "vader_neg":        vs["neg"],
        "vader_neu":        vs["neu"],
        "vader_pos":        vs["pos"],
        "vader_compound":   vs["compound"],
        "siebert_label":    sb["label"],
        "siebert_score":    sb["score"],
        "siebert_compound": compound,
    }


def evaluate():
    print(f"Loading SiEBERT ({SIEBERT_MODEL}) …")
    vader = SentimentIntensityAnalyzer()
    siebert = pipeline("text-classification", model=SIEBERT_MODEL)

    rows = []

    for dataset in DATASETS_TO_EVAL:
        dataset_dir = NARRATIVES_DIR / dataset
        if not dataset_dir.exists():
            print(f"No narratives for '{dataset}', skipping.")
            continue

        # override_pa has an extra label subfolder so go two levels deep for it
        conditions = []
        for d in sorted(dataset_dir.iterdir()):
            if not d.is_dir():
                continue
            has_provider_children = any(
                list(sub.glob("*/instance_*.json"))
                for sub in d.iterdir() if sub.is_dir()
            )
            if has_provider_children:
                conditions.append(d.name)
            else:
                for label_dir in sorted(d.iterdir()):
                    if label_dir.is_dir():
                        conditions.append(f"{d.name}/{label_dir.name}")
        if CONDITIONS_TO_EVAL:
            conditions = [c for c in conditions if c in CONDITIONS_TO_EVAL]

        for condition in conditions:
            for provider_dir in sorted((dataset_dir / condition).iterdir()):
                if not provider_dir.is_dir():
                    continue
                for model_dir in sorted(provider_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue

                    for narrative_file in sorted(model_dir.glob("instance_*.json")):
                        data = load_json(narrative_file)
                        if data.get("status") != "success" or not data.get("narrative"):
                            continue

                        idx = narrative_file.stem.split("_")[-1]
                        scores = analyse(data["narrative"], vader, siebert)
                        rows.append({
                            "dataset":           dataset,
                            "condition":         condition,
                            "narrative_provider": provider_dir.name,
                            "model":             model_dir.name,
                            "instance_idx":      idx,
                            **scores,
                        })

    if not rows:
        print("No narratives found. Check that narratives exist under results/narratives/.")
        return

    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE.relative_to(ROOT)}\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))


if __name__ == "__main__":
    evaluate()
