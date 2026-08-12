#!/usr/bin/env python
"""
Feature coverage script with demographic stratification.

For each narrative (extraction), computes:
  - shap_mentioned:       top SHAP features mentioned (mentioned=1 in features list)
  - shap_values_given:    top SHAP features with a non-NaN value (in most_important_features)
  - other_mentioned:      non-SHAP, non-protected features mentioned
  - other_values_given:   non-SHAP, non-protected features mentioned with a non-NaN value
  - protected_mentioned:  protected attributes mentioned
  - protected_values_given: protected attributes mentioned with a non-NaN value

Results are averaged per (dataset, condition, narrative_provider, extractor_provider, demographic_attribute, demographic_value)
and saved to results/fairness_eval/feature_coverage.csv.

For numeric protected attributes (age), the median is used as a cutoff: '<median' and '>=median'.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
DATASETS_TO_EVAL = ["credit"]          # or ["credit", "law", "saudi", "student"]
CONDITIONS_TO_EVAL = None             # None = all conditions found on disk
NARRATIVE_PROVIDERS_TO_EVAL = None    # None = all found on disk
EXTRACTOR_PROVIDERS_TO_EVAL = ["grok"]    # None = all found on disk
# ============================================================

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
ADVERSE_DATA_PATH = ROOT / "datasets_prep" / "data"
OUTPUT_FILE = ROOT / "results" / "fairness_eval" / "feature_coverage.csv"


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

EXTRACTIONS_DIR = ROOT / "results" / "extractions"
OUTPUT_FILE = ROOT / "results" / "fairness_eval" / "feature_coverage.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_nan(val):
    return val is None or str(val).strip().lower() in ("nan", "")


def compute_coverage(extraction, protected_attrs):
    """Return coverage counts for one extraction instance."""
    shap_names = {f["name"] for f in extraction.get("most_important_features", [])}

    shap_mentioned = 0
    shap_values_given = 0
    other_mentioned = 0
    other_values_given = 0
    protected_mentioned = 0
    protected_values_given = 0

    # shap_values_given comes from most_important_features (value field)
    for feat in extraction.get("most_important_features", []):
        if not is_nan(feat.get("value")):
            shap_values_given += 1

    # mention and value counts from features list
    for feat in extraction.get("features", []):
        name = feat.get("name")
        mentioned = feat.get("mentioned", 0) == 1
        has_value = mentioned and not is_nan(feat.get("value"))

        if name in shap_names:
            shap_mentioned += int(mentioned)
        elif name in protected_attrs:
            protected_mentioned += int(mentioned)
            protected_values_given += int(has_value)
        else:
            other_mentioned += int(mentioned)
            other_values_given += int(has_value)

    return {
        "shap_mentioned": shap_mentioned,
        "shap_values_given": shap_values_given,
        "other_mentioned": other_mentioned,
        "other_values_given": other_values_given,
        "protected_mentioned": protected_mentioned,
        "protected_values_given": protected_values_given,
    }


def evaluate():
    rows = []

    for dataset in DATASETS_TO_EVAL:
        extractions_base = EXTRACTIONS_DIR / dataset
        if not extractions_base.exists():
            print(f"No extractions found for dataset '{dataset}', skipping.")
            continue

        # Load demographic data
        demo_df = load_demographic_data(dataset)
        protected_attrs = PROTECTED_ATTRS.get(dataset, [])
        numeric_attrs = NUMERIC_ATTRS.get(dataset, [])
        demographics = get_demographic_groups(dataset, demo_df, protected_attrs, numeric_attrs) if not demo_df.empty else {}

        # override_pa has an extra label subfolder so go two levels deep for it
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

                    # Track overall and per-demographic coverage
                    overall_totals = {k: 0 for k in ("shap_mentioned", "shap_values_given",
                                                       "other_mentioned", "other_values_given",
                                                       "protected_mentioned", "protected_values_given")}
                    overall_n_instances = 0
                    
                    # demographic_metrics: {attr_name: {group_value: {totals}}}
                    demographic_totals = {}

                    for ext_file in sorted(extractor_provider_dir.glob("instance_*.json")):
                        instance_idx = int(ext_file.stem.split("_")[1])
                        extraction = load_json(ext_file)
                        coverage = compute_coverage(extraction, protected_attrs)
                        
                        # Add to overall
                        for k in overall_totals:
                            overall_totals[k] += coverage[k]
                        overall_n_instances += 1
                        
                        # Add to demographic groups
                        if instance_idx in demographics:
                            for attr, group_val in demographics[instance_idx].items():
                                if attr not in demographic_totals:
                                    demographic_totals[attr] = {}
                                if group_val not in demographic_totals[attr]:
                                    demographic_totals[attr][group_val] = {k: 0 for k in ("shap_mentioned", "shap_values_given",
                                                                                            "other_mentioned", "other_values_given",
                                                                                            "protected_mentioned", "protected_values_given")}
                                    demographic_totals[attr][group_val]["n_instances"] = 0
                                
                                for k in overall_totals:
                                    demographic_totals[attr][group_val][k] += coverage[k]
                                demographic_totals[attr][group_val]["n_instances"] += 1

                    if overall_n_instances == 0:
                        continue

                    # Add overall row
                    row = {
                        "dataset": dataset,
                        "condition": condition,
                        "narrative_provider": narrative_provider,
                        "extractor_provider": extractor_provider,
                        "demographic_attribute": "OVERALL",
                        "demographic_value": "ALL",
                        "n_instances": overall_n_instances,
                        "avg_shap_mentioned": overall_totals["shap_mentioned"] / overall_n_instances,
                        "avg_shap_values_given": overall_totals["shap_values_given"] / overall_n_instances,
                        "avg_other_mentioned": overall_totals["other_mentioned"] / overall_n_instances,
                        "avg_other_values_given": overall_totals["other_values_given"] / overall_n_instances,
                        "avg_protected_mentioned": overall_totals["protected_mentioned"] / overall_n_instances,
                        "avg_protected_values_given": overall_totals["protected_values_given"] / overall_n_instances,
                    }
                    rows.append(row)
                    
                    # Add demographic group rows
                    for attr in sorted(demographic_totals.keys()):
                        for group_val in sorted(demographic_totals[attr].keys()):
                            metrics = demographic_totals[attr][group_val]
                            n_instances = metrics["n_instances"]
                            row = {
                                "dataset": dataset,
                                "condition": condition,
                                "narrative_provider": narrative_provider,
                                "extractor_provider": extractor_provider,
                                "demographic_attribute": attr,
                                "demographic_value": group_val,
                                "n_instances": n_instances,
                                "avg_shap_mentioned": metrics["shap_mentioned"] / n_instances,
                                "avg_shap_values_given": metrics["shap_values_given"] / n_instances,
                                "avg_other_mentioned": metrics["other_mentioned"] / n_instances,
                                "avg_other_values_given": metrics["other_values_given"] / n_instances,
                                "avg_protected_mentioned": metrics["protected_mentioned"] / n_instances,
                                "avg_protected_values_given": metrics["protected_values_given"] / n_instances,
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
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df.to_string(index=False))


if __name__ == "__main__":
    evaluate()
