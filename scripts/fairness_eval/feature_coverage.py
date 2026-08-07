#!/usr/bin/env python
"""
Feature coverage script.

For each narrative (extraction), computes:
  - shap_mentioned:       top SHAP features mentioned (mentioned=1 in features list)
  - shap_values_given:    top SHAP features with a non-NaN value (in most_important_features)
  - other_mentioned:      non-SHAP, non-protected features mentioned
  - other_values_given:   non-SHAP, non-protected features mentioned with a non-NaN value
  - protected_mentioned:  protected attributes mentioned
  - protected_values_given: protected attributes mentioned with a non-NaN value

Results are averaged per (dataset, condition, narrative_provider, extractor_provider)
and saved to results/fairness_eval/feature_coverage.csv.
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
DATASETS_TO_EVAL = ["law"]          # or ["credit", "law", "saudi", "student"]
CONDITIONS_TO_EVAL = None             # None = all conditions found on disk
EXTRACTOR_PROVIDERS_TO_EVAL = ["majority_voted"]    # None = all found on disk
# ============================================================

PROTECTED_ATTRS = {
    "credit":  ["age", "sex", "foreign_worker"],
    "law":     ["gender", "race"],
    "saudi":   ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
}

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

        protected_attrs = PROTECTED_ATTRS.get(dataset, [])

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

                for extractor_provider_dir in sorted(narrative_provider_dir.iterdir()):
                    if not extractor_provider_dir.is_dir():
                        continue
                    extractor_provider = extractor_provider_dir.name

                    if EXTRACTOR_PROVIDERS_TO_EVAL and extractor_provider not in EXTRACTOR_PROVIDERS_TO_EVAL:
                        continue

                    totals = {k: 0 for k in ("shap_mentioned", "shap_values_given",
                                             "other_mentioned", "other_values_given",
                                             "protected_mentioned", "protected_values_given")}
                    n_instances = 0

                    for ext_file in sorted(extractor_provider_dir.glob("instance_*.json")):
                        extraction = load_json(ext_file)
                        coverage = compute_coverage(extraction, protected_attrs)
                        for k in totals:
                            totals[k] += coverage[k]
                        n_instances += 1

                    if n_instances == 0:
                        continue

                    rows.append({
                        "dataset": dataset,
                        "condition": condition,
                        "narrative_provider": narrative_provider,
                        "extractor_provider": extractor_provider,
                        "n_instances": n_instances,
                        "avg_shap_mentioned": totals["shap_mentioned"] / n_instances,
                        "avg_shap_values_given": totals["shap_values_given"] / n_instances,
                        "avg_other_mentioned": totals["other_mentioned"] / n_instances,
                        "avg_other_values_given": totals["other_values_given"] / n_instances,
                        "avg_protected_mentioned": totals["protected_mentioned"] / n_instances,
                        "avg_protected_values_given": totals["protected_values_given"] / n_instances,
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
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df.to_string(index=False))


if __name__ == "__main__":
    evaluate()
