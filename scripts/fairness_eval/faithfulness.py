#!/usr/bin/env python
"""
Faithfulness evaluation script with demographic stratification.

Computes metrics comparing LLM-extracted narrative information against SHAP ground truth:
  - Rank accuracy (per rank 1/2/3): correct feature name at each rank position
  - Sign accuracy: correct directional influence sign
  - Value accuracy: % of mentioned feature values that match ground truth

Results are aggregated per (dataset, condition, narrative_provider, extractor_provider, demographic_attribute, demographic_value)
and saved to results/fairness_eval/faithfulness.csv.

For numeric protected attributes (age), the median is used as a cutoff: '<median' and '>=median'.
For categorical protected attributes, the actual values are used.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
DATASETS_TO_EVAL = ["credit", "law", "saudi", "student"]            # or ["credit", "law", "saudi", "student"]
CONDITIONS_TO_EVAL = ["include_pa"]            # None = all conditions found on disk; or e.g. ["include_pa", "exclude_pa", "override_pa/gender_female__race_black"]
NARRATIVE_PROVIDERS_TO_EVAL = ["grok", "openai", "deepseek"]  # None = all found on disk
EXTRACTOR_PROVIDERS_TO_EVAL = ["grok"]    # None = all found on disk
# ============================================================

# Protected attributes per dataset (excluded from model training)
PROTECTED_ATTRS = {
    "credit":  ["age", "sex", "foreign_worker"],
    "law":     ["gender", "race"],
    "saudi":   ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
}

# Numeric attributes that should be binned by median
NUMERIC_ATTRS = {
    "credit":  ["age"],
    "law":     [],
    "saudi":   ["Age"],
    "student": ["age"],
}

EXTRACTIONS_DIR = ROOT / "results" / "extractions"
GT_DIR = ROOT / "results" / "ground_truth" / "json"
OUTPUT_FILE = ROOT / "results" / "fairness_eval" / "faithfulness.csv"
ADVERSE_DATA_PATH = ROOT / "datasets_prep" / "data"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_demographic_data(dataset):
    """Load demographic attributes from adverse CSV."""
    adverse_file = ADVERSE_DATA_PATH / f"{dataset}_dataset" / f"{dataset}_adverse.csv"
    if not adverse_file.exists():
        print(f"Warning: {adverse_file} not found")
        return {}
    
    df = pd.read_csv(adverse_file)
    return df


def get_demographic_groups(dataset, df, protected_attrs, numeric_attrs):
    """
    Create demographic groups for each protected attribute.
    Returns dict: {instance_idx: {attr_name: group_value}}
    
    For numeric attributes, groups are '<median' and '>=median'.
    For categorical attributes, groups are the actual values.
    """
    demographics = {}
    
    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        
        if attr in numeric_attrs:
            # Numeric: bin by median
            median_val = df[attr].median()
            for idx, row in df.iterrows():
                val = row[attr]
                group = "<median" if val < median_val else ">=median"
                if idx not in demographics:
                    demographics[idx] = {}
                demographics[idx][attr] = group
        else:
            # Categorical: use value directly
            for idx, row in df.iterrows():
                val = str(row[attr])
                if idx not in demographics:
                    demographics[idx] = {}
                demographics[idx][attr] = val
    
    return demographics


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

        # Load demographic data
        demo_df = load_demographic_data(dataset)
        protected_attrs = PROTECTED_ATTRS.get(dataset, [])
        numeric_attrs = NUMERIC_ATTRS.get(dataset, [])
        demographics = get_demographic_groups(dataset, demo_df, protected_attrs, numeric_attrs) if not demo_df.empty else {}

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

                if NARRATIVE_PROVIDERS_TO_EVAL and narrative_provider not in NARRATIVE_PROVIDERS_TO_EVAL:
                    continue

                for extractor_provider_dir in sorted(narrative_provider_dir.iterdir()):
                    if not extractor_provider_dir.is_dir():
                        continue
                    extractor_provider = extractor_provider_dir.name

                    if EXTRACTOR_PROVIDERS_TO_EVAL and extractor_provider not in EXTRACTOR_PROVIDERS_TO_EVAL:
                        continue

                    # Collect instances and compute metrics by demographic group
                    # Structure: {attr_name: {group_value: {metrics}}}
                    demographic_metrics = {}
                    overall_metrics = {
                        "rank_correct": {1: 0, 2: 0, 3: 0},
                        "rank_total": {1: 0, 2: 0, 3: 0},
                        "sign_correct": 0,
                        "sign_total": 0,
                        "val_counts": {"shap": [0, 0], "protected": [0, 0], "other": [0, 0], "all": [0, 0]},
                        "prob_correct": 0,
                        "prob_total": 0,
                        "n_instances": 0,
                        "instances": []
                    }

                    for ext_file in sorted(extractor_provider_dir.glob("instance_*.json")):
                        instance_idx = int(ext_file.stem.split("_")[1])
                        gt_file = gt_base / f"instance_{instance_idx}.json"

                        if not gt_file.exists():
                            print(f"  Warning: no ground truth for {dataset} instance {instance_idx}, skipping.")
                            continue

                        gt = load_json(gt_file)
                        extraction = load_json(ext_file)
                        
                        # Compute metrics for this instance
                        rank_acc = compute_rank_accuracy(gt, extraction)
                        sign_acc = compute_sign_accuracy(gt, extraction)
                        val_acc = compute_value_accuracy(gt, extraction, protected_attrs)
                        
                        try:
                            gt_prob = float(gt.get("predicted_probability", "NaN"))
                            ext_prob = float(extraction.get("predicted_probability", "NaN"))
                            prob_match = int(gt_prob == ext_prob)
                        except (TypeError, ValueError):
                            prob_match = None
                        
                        # Update overall metrics
                        for r, correct in rank_acc.items():
                            overall_metrics["rank_correct"][r] += int(correct)
                            overall_metrics["rank_total"][r] += 1
                        
                        sc, st = sign_acc
                        overall_metrics["sign_correct"] += sc
                        overall_metrics["sign_total"] += st
                        
                        for cat, (c, t) in val_acc.items():
                            overall_metrics["val_counts"][cat][0] += c
                            overall_metrics["val_counts"][cat][1] += t
                        
                        if prob_match is not None:
                            overall_metrics["prob_correct"] += prob_match
                            overall_metrics["prob_total"] += 1
                        
                        overall_metrics["n_instances"] += 1
                        overall_metrics["instances"].append((instance_idx, rank_acc, sign_acc, val_acc, prob_match))
                        
                        # Update demographic group metrics
                        if instance_idx in demographics:
                            for attr, group_val in demographics[instance_idx].items():
                                if attr not in demographic_metrics:
                                    demographic_metrics[attr] = {}
                                if group_val not in demographic_metrics[attr]:
                                    demographic_metrics[attr][group_val] = {
                                        "rank_correct": {1: 0, 2: 0, 3: 0},
                                        "rank_total": {1: 0, 2: 0, 3: 0},
                                        "sign_correct": 0,
                                        "sign_total": 0,
                                        "val_counts": {"shap": [0, 0], "protected": [0, 0], "other": [0, 0], "all": [0, 0]},
                                        "prob_correct": 0,
                                        "prob_total": 0,
                                        "n_instances": 0,
                                    }
                                
                                metrics = demographic_metrics[attr][group_val]
                                for r, correct in rank_acc.items():
                                    metrics["rank_correct"][r] += int(correct)
                                    metrics["rank_total"][r] += 1
                                
                                sc, st = sign_acc
                                metrics["sign_correct"] += sc
                                metrics["sign_total"] += st
                                
                                for cat, (c, t) in val_acc.items():
                                    metrics["val_counts"][cat][0] += c
                                    metrics["val_counts"][cat][1] += t
                                
                                if prob_match is not None:
                                    metrics["prob_correct"] += prob_match
                                    metrics["prob_total"] += 1
                                
                                metrics["n_instances"] += 1

                    if overall_metrics["n_instances"] == 0:
                        continue

                    # Add overall row
                    row = {
                        "dataset": dataset,
                        "condition": condition,
                        "narrative_provider": narrative_provider,
                        "extractor_provider": extractor_provider,
                        "demographic_attribute": "OVERALL",
                        "demographic_value": "ALL",
                        "n_instances": overall_metrics["n_instances"],
                        "rank1_accuracy": overall_metrics["rank_correct"][1] / overall_metrics["rank_total"][1] if overall_metrics["rank_total"][1] else None,
                        "rank2_accuracy": overall_metrics["rank_correct"][2] / overall_metrics["rank_total"][2] if overall_metrics["rank_total"][2] else None,
                        "rank3_accuracy": overall_metrics["rank_correct"][3] / overall_metrics["rank_total"][3] if overall_metrics["rank_total"][3] else None,
                        "rank_total_accuracy": sum(overall_metrics["rank_correct"].values()) / sum(overall_metrics["rank_total"].values()) if sum(overall_metrics["rank_total"].values()) else None,
                        "sign_accuracy": overall_metrics["sign_correct"] / overall_metrics["sign_total"] if overall_metrics["sign_total"] else None,
                        "shap_value_accuracy": overall_metrics["val_counts"]["shap"][0] / overall_metrics["val_counts"]["shap"][1] if overall_metrics["val_counts"]["shap"][1] else None,
                        "protected_value_accuracy": overall_metrics["val_counts"]["protected"][0] / overall_metrics["val_counts"]["protected"][1] if overall_metrics["val_counts"]["protected"][1] else None,
                        "other_value_accuracy": overall_metrics["val_counts"]["other"][0] / overall_metrics["val_counts"]["other"][1] if overall_metrics["val_counts"]["other"][1] else None,
                        "all_value_accuracy": overall_metrics["val_counts"]["all"][0] / overall_metrics["val_counts"]["all"][1] if overall_metrics["val_counts"]["all"][1] else None,
                        "predicted_probability_accuracy": overall_metrics["prob_correct"] / overall_metrics["prob_total"] if overall_metrics["prob_total"] else None,
                    }
                    rows.append(row)
                    
                    # Add demographic group rows
                    for attr in sorted(demographic_metrics.keys()):
                        for group_val in sorted(demographic_metrics[attr].keys()):
                            metrics = demographic_metrics[attr][group_val]
                            row = {
                                "dataset": dataset,
                                "condition": condition,
                                "narrative_provider": narrative_provider,
                                "extractor_provider": extractor_provider,
                                "demographic_attribute": attr,
                                "demographic_value": group_val,
                                "n_instances": metrics["n_instances"],
                                "rank1_accuracy": metrics["rank_correct"][1] / metrics["rank_total"][1] if metrics["rank_total"][1] else None,
                                "rank2_accuracy": metrics["rank_correct"][2] / metrics["rank_total"][2] if metrics["rank_total"][2] else None,
                                "rank3_accuracy": metrics["rank_correct"][3] / metrics["rank_total"][3] if metrics["rank_total"][3] else None,
                                "rank_total_accuracy": sum(metrics["rank_correct"].values()) / sum(metrics["rank_total"].values()) if sum(metrics["rank_total"].values()) else None,
                                "sign_accuracy": metrics["sign_correct"] / metrics["sign_total"] if metrics["sign_total"] else None,
                                "shap_value_accuracy": metrics["val_counts"]["shap"][0] / metrics["val_counts"]["shap"][1] if metrics["val_counts"]["shap"][1] else None,
                                "protected_value_accuracy": metrics["val_counts"]["protected"][0] / metrics["val_counts"]["protected"][1] if metrics["val_counts"]["protected"][1] else None,
                                "other_value_accuracy": metrics["val_counts"]["other"][0] / metrics["val_counts"]["other"][1] if metrics["val_counts"]["other"][1] else None,
                                "all_value_accuracy": metrics["val_counts"]["all"][0] / metrics["val_counts"]["all"][1] if metrics["val_counts"]["all"][1] else None,
                                "predicted_probability_accuracy": metrics["prob_correct"] / metrics["prob_total"] if metrics["prob_total"] else None,
                            }
                            rows.append(row)

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
