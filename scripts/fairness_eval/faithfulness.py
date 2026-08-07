#!/usr/bin/env python
"""
Faithfulness evaluation script.

Computes three metrics comparing LLM-extracted narrative information against SHAP ground truth:
  - Rank accuracy (per rank 1/2/3): correct feature name at each rank position
  - Sign accuracy (per rank 1/2/3): correct directional influence sign at each rank
  - Value accuracy: % of mentioned feature values that match ground truth

Results are aggregated per (dataset, condition, narrative_provider, extractor_provider)
and saved to results/fairness_eval/faithfulness.csv.
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
DATASETS_TO_EVAL = ["law"]            # or ["credit", "law", "saudi", "student"]
CONDITIONS_TO_EVAL = None             # None = all conditions found on disk; or e.g. ["include_pa", "exclude_pa", "override_pa/gender_female__race_black"]
EXTRACTOR_PROVIDERS_TO_EVAL = ["majority_voted"]    # None = all found on disk
# ============================================================

# Protected attributes per dataset (excluded from model training)
PROTECTED_ATTRS = {
    "credit":  ["age", "sex", "foreign_worker"],
    "law":     ["gender", "race"],
    "saudi":   ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
}

EXTRACTIONS_DIR = ROOT / "results" / "extractions"
GT_DIR = ROOT / "results" / "ground_truth" / "json"
OUTPUT_FILE = ROOT / "results" / "fairness_eval" / "faithfulness.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gt_value_lookup(gt):
    """Build {feature_name: gt_value} from both top features and other features."""
    lookup = {}
    for feat in gt.get("most_important_features", []):
        lookup[feat["name"]] = feat["value"]
    for feat in gt.get("features", []):
        lookup[feat["name"]] = feat["value"]
    return lookup


def compute_rank_accuracy(gt, extraction):
    """Return {rank: bool} — whether the extracted feature name matches GT at each rank."""
    gt_by_rank = {f["rank"]: f["name"] for f in gt.get("most_important_features", [])}
    ext_by_rank = {f["rank"]: f["name"] for f in extraction.get("most_important_features", [])}
    return {r: (ext_by_rank.get(r) == gt_by_rank.get(r)) for r in [1, 2, 3]}


def compute_sign_accuracy(gt, extraction):
    """Return (correct, total) matching signs by feature NAME across extracted top features."""
    gt_by_name = {f["name"]: int(f["sign"]) for f in gt.get("most_important_features", [])}
    correct = 0
    total = 0
    for feat in extraction.get("most_important_features", []):
        name = feat.get("name")
        if name not in gt_by_name:
            continue
        try:
            correct += int(int(feat["sign"]) == gt_by_name[name])
        except (ValueError, TypeError):
            pass
        total += 1
    return correct, total


def compute_value_accuracy(gt, extraction, protected_attrs):
    """
    Return dict with (correct, total) for shap, protected, other, and all feature values.
    Only counts features where mentioned=1 and extracted value is not NaN/None.
    """
    gt_shap_names = {f["name"] for f in gt.get("most_important_features", [])}
    gt_lookup = build_gt_value_lookup(gt)

    counts = {
        "shap": [0, 0], "protected": [0, 0],
        "other": [0, 0], "all": [0, 0]
    }

    for feat in extraction.get("features", []):
        if feat.get("mentioned") != 1:
            continue
        ext_val = feat.get("value")
        if ext_val is None or str(ext_val).strip().lower() in ("nan", ""):
            continue
        name = feat["name"]
        if name not in gt_lookup:
            continue
        gt_val = gt_lookup[name]
        if gt_val is None or str(gt_val).strip().lower() in ("nan", ""):
            continue
        try:
            match = int(float(ext_val) == float(gt_val))
        except (ValueError, TypeError):
            match = int(str(ext_val).strip() == str(gt_val).strip())

        if name in gt_shap_names:
            category = "shap"
        elif name in protected_attrs:
            category = "protected"
        else:
            category = "other"

        counts[category][0] += match
        counts[category][1] += 1
        counts["all"][0] += match
        counts["all"][1] += 1

    return counts


def evaluate():
    rows = []

    for dataset in DATASETS_TO_EVAL:
        extractions_base = EXTRACTIONS_DIR / dataset
        gt_base = GT_DIR / dataset

        if not extractions_base.exists():
            print(f"No extractions found for dataset '{dataset}', skipping.")
            continue

        # Discover conditions on disk; override_pa has an extra label subfolder so go two levels deep for it
        conditions = []
        for d in sorted(extractions_base.iterdir()):
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
            condition_dir = extractions_base / condition

            for narrative_provider_dir in sorted(condition_dir.iterdir()):
                if not narrative_provider_dir.is_dir():
                    continue
                narrative_provider = narrative_provider_dir.name

                for extractor_provider_dir in sorted(narrative_provider_dir.iterdir()):
                    if not extractor_provider_dir.is_dir():
                        continue
                    extractor_provider = extractor_provider_dir.name

                    if EXTRACTOR_PROVIDERS_TO_EVAL and extractor_provider not in EXTRACTOR_PROVIDERS_TO_EVAL:
                        continue

                    protected_attrs = PROTECTED_ATTRS.get(dataset, [])
                    rank_correct = {1: 0, 2: 0, 3: 0}
                    rank_total = {1: 0, 2: 0, 3: 0}
                    sign_correct = 0
                    sign_total = 0
                    val_counts = {"shap": [0, 0], "protected": [0, 0], "other": [0, 0], "all": [0, 0]}
                    prob_correct = 0  # exact matches for predicted_probability
                    prob_total = 0
                    n_instances = 0

                    for ext_file in sorted(extractor_provider_dir.glob("instance_*.json")):
                        instance_idx = int(ext_file.stem.split("_")[1])
                        gt_file = gt_base / f"instance_{instance_idx}.json"

                        if not gt_file.exists():
                            print(f"  Warning: no ground truth for {dataset} instance {instance_idx}, skipping.")
                            continue

                        gt = load_json(gt_file)
                        extraction = load_json(ext_file)
                        n_instances += 1

                        for r, correct in compute_rank_accuracy(gt, extraction).items():
                            rank_correct[r] += int(correct)
                            rank_total[r] += 1

                        sc, st = compute_sign_accuracy(gt, extraction)
                        sign_correct += sc
                        sign_total += st

                        for cat, (c, t) in compute_value_accuracy(gt, extraction, protected_attrs).items():
                            val_counts[cat][0] += c
                            val_counts[cat][1] += t

                        try:
                            gt_prob = float(gt.get("predicted_probability", "NaN"))
                            ext_prob = float(extraction.get("predicted_probability", "NaN"))
                            prob_correct += int(gt_prob == ext_prob)
                            prob_total += 1
                        except (TypeError, ValueError):
                            pass

                    if n_instances == 0:
                        continue

                    rows.append({
                        "dataset": dataset,
                        "condition": condition,
                        "narrative_provider": narrative_provider,
                        "extractor_provider": extractor_provider,
                        "n_instances": n_instances,
                        "rank1_accuracy": rank_correct[1] / rank_total[1] if rank_total[1] else None,
                        "rank2_accuracy": rank_correct[2] / rank_total[2] if rank_total[2] else None,
                        "rank3_accuracy": rank_correct[3] / rank_total[3] if rank_total[3] else None,
                        "rank_total_accuracy": sum(rank_correct.values()) / sum(rank_total.values()) if sum(rank_total.values()) else None,
                        "sign_accuracy": sign_correct / sign_total if sign_total else None,
                        "shap_value_accuracy": val_counts["shap"][0] / val_counts["shap"][1] if val_counts["shap"][1] else None,
                        "protected_value_accuracy": val_counts["protected"][0] / val_counts["protected"][1] if val_counts["protected"][1] else None,
                        "other_value_accuracy": val_counts["other"][0] / val_counts["other"][1] if val_counts["other"][1] else None,
                        "all_value_accuracy": val_counts["all"][0] / val_counts["all"][1] if val_counts["all"][1] else None,
                        "predicted_probability_accuracy": prob_correct / prob_total if prob_total else None,
                    })

    if not rows:
        print("No results found. Check that extractions exist under results/extractions/.")
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
