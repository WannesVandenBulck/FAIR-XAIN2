#!/usr/bin/env python
"""
Surface properties script.

For each narrative, computes linguistic surface metrics:
  - word_count:            total number of words
  - sentence_count:        total number of sentences
  - avg_sentence_length:   mean words per sentence
  - type_token_ratio:      unique words / total words (lexical diversity)
  - dist2:                 unique bigrams / total bigrams (bigram diversity)
  - connectives_ratio:     discourse connectives (Das et al. 142-term PDTB lexicon) / total words
  - cause_effect_ratio:    causal markers (19-term list) / total words
  - verb_ratio:            verbs (NLTK POS VB*) / total words
  - flesch_kincaid_grade:  Flesch-Kincaid Grade Level (higher = harder to read)
  - dale_chall_score:      Dale-Chall Readability Score (higher = harder to read)

Results are averaged per (dataset, condition, narrative_provider, model)
and saved to results/fairness_eval/surface_properties.csv.
"""

import json
import re
from pathlib import Path
import pandas as pd
import textstat
import nltk
from nltk.util import bigrams as nltk_bigrams

nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Connectives: exact 142-term en_dimlex lexicon (Das et al.) ──────────────
# Source: https://github.com/discourse-lab/en_dimlex/blob/master/en_dimlex.xml
# Multi-word phrases come first (sorted longest→shortest) so that phrase
# matching runs before any single-word substring could be counted separately.
CONNECTIVES = [
    # ── multi-word phrases (longest first) ───────────────────────────────────
    "on the one hand",                     # id 100 (assumes "on the other hand" always follows)
    "quite the contrary",                  # id 109
    "as an alternative",                   # id  99
    "on the other hand",                   # id  28
    "at the same time",                    # id 131
    "before and after",                    # id  65
    "on the contrary",                     # id  92
    "irrespective of",                     # id 137
    "in response to",                      # id 126
    "in other words",                      # id  75
    "in addition to",                      # id 132
    "as a result of",                      # id 136
    "for one thing",                       # id 111
    "at that point",                       # id 124
    "by comparison",                       # id  57
    "in particular",                       # id  46
    "for instance",                        # id  30
    "in any event",                        # id 121
    "in any case",                         # id 134
    "in spite of",                         # id 133
    "except that",                         # id 129
    "when and if",                         # id  96
    "even though",                         # id 116
    "neither nor",                         # id  79
    "if and when",                         # id  52
    "at the time",                         # id 106
    "for example",                         # id   8
    "in this way",                         # id 107
    "in addition",                         # id  26
    "in contrast",                         # id  59
    "by contrast",                         # id  68
    "as a result",                         # id  41
    "rather than",                         # id 138
    "as long as",                          # id  17
    "in the end",                          # id  71
    "after that",                          # id 139
    "insofar as",                          # id  97
    "in essence",                          # id 125
    "as soon as",                          # id  76
    "because of",                          # id 101
    "aside from",                          # id 102
    "given that",                          # id 120
    "by the way",                          # id 119
    "instead of",                          # id 113
    "after all",                           # id 135
    "as though",                           # id  82
    "either or",                           # id  42
    "not only",                            # id 140  (from "not only/but")
    "in short",                            # id  78
    "now that",                            # id  64
    "so that",                             # id  23
    "much as",                             # id  87
    "as well",                             # id  86
    "in fact",                             # id  37
    "in turn",                             # id  44
    "in case",                             # id 114
    "by then",                             # id  90
    "for one",                             # id 103
    "even so",                             # id 108
    "even if",                             # id 118
    "if then",                             # id  69
    "in sum",                              # id  95
    "as if",                               # id  62
    # ── single-word terms (in XML id order) ──────────────────────────────────
    "once",           # id   1
    "although",       # id   2
    "though",         # id   3
    "but",            # id   4
    "because",        # id   5
    "nevertheless",   # id   6
    "before",         # id   7
    "until",          # id   9
    "if",             # id  10
    "previously",     # id  11
    "when",           # id  12
    "and",            # id  13
    "so",             # id  14
    "then",           # id  15
    "while",          # id  16
    "however",        # id  18
    "also",           # id  19
    "after",          # id  20
    "separately",     # id  21
    "still",          # id  22
    "or",             # id  24
    "moreover",       # id  25
    "instead",        # id  27
    "as",             # id  29
    "nonetheless",    # id  31
    "unless",         # id  32
    "meanwhile",      # id  33
    "yet",            # id  34
    "since",          # id  35
    "rather",         # id  36
    "indeed",         # id  38
    "later",          # id  39
    "ultimately",     # id  40
    "therefore",      # id  43
    "thus",           # id  45
    "further",        # id  47
    "afterward",      # id  48
    "next",           # id  49
    "similarly",      # id  50
    "besides",        # id  51
    "nor",            # id  53
    "alternatively",  # id  54
    "whereas",        # id  55
    "overall",        # id  56
    "till",           # id  58
    "finally",        # id  60
    "otherwise",      # id  61
    "thereby",        # id  63
    "additionally",   # id  66
    "meantime",       # id  67
    "likewise",       # id  70
    "regardless",     # id  72
    "thereafter",     # id  73
    "earlier",        # id  74
    "except",         # id  77
    "furthermore",    # id  80
    "lest",           # id  81
    "specifically",   # id  83
    "conversely",     # id  84
    "consequently",   # id  85
    "plus",           # id  88
    "hence",          # id  89
    "accordingly",    # id  91
    "simultaneously", # id  93
    "for",            # id  94
    "else",           # id  98
    "whatever",       # id 104
    #"when/then",      # id 105  (non-standard form; contributes 0 matches)
    "everytime",      # id 110
    "despite",        # id 112
    "without",        # id 115
    "with",           # id 117
    "essentially",    # id 122
    "given",          # id 123
    "anyway",         # id 127
    "upon",           # id 128
    "eventually",     # id 130
    "whenever",       # id 141
    "particularly",   # id 142
]

# ── Causal markers: 19-term list (Appendix A) ────────────────────────────────
CAUSAL_MARKERS = [
    "as a result of", "as a result", "because of", "given that",
    "in response to", "in this way", "in turn", "so that",
    "accordingly", "because", "consequently", "due to",
    "given", "hence", "since", "so", "thereby", "therefore", "thus",
]

ROOT = Path(__file__).parent.parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
DATASETS_TO_EVAL = ["law"]      # or ["credit", "law", "saudi", "student"]
CONDITIONS_TO_EVAL = None       # None = all conditions found on disk
# ============================================================

NARRATIVES_DIR = ROOT / "results" / "narratives"
OUTPUT_FILE = ROOT / "results" / "fairness_eval" / "surface_properties.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_phrase_matches(text_lower, phrases):
    """Count non-overlapping occurrences of phrases using word-boundary-aware matching."""
    count = 0
    for phrase in phrases:
        # use word boundaries for single-word phrases, plain search for multi-word
        if " " in phrase:
            count += text_lower.count(phrase)
        else:
            count += len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_lower))
    return count


def compute_surface_metrics(text):
    """Return surface linguistic metrics for a single narrative text."""
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    sentence_count = textstat.sentence_count(text)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    ttr = len(set(words)) / word_count if word_count else 0

    # Dist2: unique bigrams / total bigrams
    all_bigrams = list(nltk_bigrams(words))
    dist2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0

    # Connectives Ratio and Cause-Effect Ratio
    text_lower = text.lower()
    cr = count_phrase_matches(text_lower, CONNECTIVES) / word_count if word_count else 0
    cer = count_phrase_matches(text_lower, CAUSAL_MARKERS) / word_count if word_count else 0

    # Verb Ratio via NLTK POS tagging
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    verb_count = sum(1 for _, tag in tagged if tag.startswith("VB"))
    vr = verb_count / word_count if word_count else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": ttr,
        "dist2": dist2,
        "connectives_ratio": cr,
        "cause_effect_ratio": cer,
        "verb_ratio": vr,
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "dale_chall_score": textstat.dale_chall_readability_score(text),
    }


def evaluate():
    rows = []

    for dataset in DATASETS_TO_EVAL:
        dataset_dir = NARRATIVES_DIR / dataset
        if not dataset_dir.exists():
            print(f"No narratives found for dataset '{dataset}', skipping.")
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
            condition_dir = dataset_dir / condition

            for provider_dir in sorted(condition_dir.iterdir()):
                if not provider_dir.is_dir():
                    continue
                provider = provider_dir.name

                for model_dir in sorted(provider_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name

                    totals = {k: 0.0 for k in ("word_count", "sentence_count",
                                                "avg_sentence_length", "type_token_ratio",
                                                "dist2", "connectives_ratio",
                                                "cause_effect_ratio", "verb_ratio",
                                                "flesch_kincaid_grade", "dale_chall_score")}
                    n_instances = 0

                    for narrative_file in sorted(model_dir.glob("instance_*.json")):
                        data = load_json(narrative_file)
                        if data.get("status") != "success" or not data.get("narrative"):
                            continue
                        metrics = compute_surface_metrics(data["narrative"])
                        for k in totals:
                            totals[k] += metrics[k]
                        n_instances += 1

                    if n_instances == 0:
                        continue

                    rows.append({
                        "dataset": dataset,
                        "condition": condition,
                        "narrative_provider": provider,
                        "model": model,
                        "n_instances": n_instances,
                        "avg_word_count": totals["word_count"] / n_instances,
                        "avg_sentence_count": totals["sentence_count"] / n_instances,
                        "avg_sentence_length": totals["avg_sentence_length"] / n_instances,
                        "avg_type_token_ratio": totals["type_token_ratio"] / n_instances,
                        "avg_dist2": totals["dist2"] / n_instances,
                        "avg_connectives_ratio": totals["connectives_ratio"] / n_instances,
                        "avg_cause_effect_ratio": totals["cause_effect_ratio"] / n_instances,
                        "avg_verb_ratio": totals["verb_ratio"] / n_instances,
                        "avg_flesch_kincaid_grade": totals["flesch_kincaid_grade"] / n_instances,
                        "avg_dale_chall_score": totals["dale_chall_score"] / n_instances,
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
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df.to_string(index=False))


if __name__ == "__main__":
    evaluate()
