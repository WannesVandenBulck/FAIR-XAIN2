#!/usr/bin/env python
"""
Generate per_narrative_metrics_<dataset>.csv files in the exact column format
specified by the research-team template (kolommen_research.xlsx).

DESIGN DECISIONS CONFIRMED WITH THE RESEARCH TEAM (do not change without
re-confirming):

1. EXTRACTOR: only the "grok" extractor's files are used for every
   "_extracted" field, regardless of which provider generated the narrative
   or the extraction condition. This was confirmed because the grok
   extractor is the only one with complete coverage across all
   dataset x provider x condition combinations; deepseek/openai extractor
   folders exist but are incomplete or absent for some datasets.
2. `other_feature_x_value_scoring` (1/0/NaN) is included for consistency
   with the rank_N and PA_N groups, even though the original template only
   listed `_mentioned`, `_value_GT`, `_value_extracted` for other features.
3. No separate `PA{n}_value_GT` column is produced. `PA{n}_GT` already
   names the attribute (e.g. "sex"); the true per-instance value used for
   scoring is looked up internally from the ground-truth file using that
   name, but is not exposed as its own output column (per confirmation).
4. Whenever an "_extracted" field is missing (nothing extracted at that
   rank / value / attribute), the corresponding "_scoring" field is NaN,
   not 0 — a missing extraction is treated as "no comparison possible",
   not as "wrong". This applies uniformly to predicted_probability,
   rank_N, rank_N_sign, rank_N_value, PA_N_value, and other_feature_value
   scoring.
5. `rank_N_sign_extracted` and `rank_N_value_extracted` are BOTH resolved by
   searching the EXTRACTION's own output for the feature named at GT rank N
   — NOT by the position the extraction itself assigned that feature (or
   any other feature). This means `rank_N_scoring` (position/name match at
   rank N) can disagree with `rank_N_sign_scoring` and `rank_N_value_scoring`:
   a feature ranked 1st in GT but placed 2nd by the extraction scores
   rank_1_scoring = 0, while rank_1_sign_scoring and rank_1_value_scoring
   can still be 1 if the sign/value recorded for that feature (wherever the
   extraction placed or mentioned it) matches GT.
     - Sign is only recoverable from the extraction's `most_important_features`
       list (its top-3), since the extraction schema only records `sign` for
       those items; if the GT rank-N feature isn't in that list, sign is NaN.
     - Value is recoverable from EITHER the extraction's `most_important_features`
       list (always counts, since a ranked feature is inherently "mentioned")
       OR its regular `features` list (only counts if that entry's own
       `mentioned` flag is 1); if the GT rank-N feature appears in neither,
       or is present in `features` with `mentioned` = 0, value is NaN.
6. Only adversely-classified instances are included (i.e. exactly the
   instances present under results/ground_truth/json/<dataset>/), matching
   the existing per_narrative_metrics_<dataset>.csv scope.

INPUT
-----
results/ground_truth/json/<dataset>/instance_<id>.json
    {"predicted_probability": ..., "most_important_features": [...], "features": [...]}
results/extractions/<dataset>/<condition>/<provider>/grok/instance_<id>.json
    same schema as ground truth, extracted from the narrative by the grok extractor.

OUTPUT
------
results/fairness_eval/per_narrative_metrics_<dataset>.csv
    One row per instance x provider x condition.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent.parent
GT_DIR = ROOT / "results" / "ground_truth" / "json"
EXTRACTIONS_DIR = ROOT / "results" / "extractions"
OUTPUT_DIR = ROOT / "results" / "fairness_eval"

DATASETS = ["credit", "saudi", "student", "law"]
NARRATIVE_PROVIDERS = ["grok", "openai", "deepseek"]
EXTRACTOR = "grok"  # fixed extractor for all "_extracted" fields (see decision 1)

# condition directory name on disk -> value written to the "condition" column
CONDITIONS = {
    "include_pa": "pa_included",
    "exclude_pa": "pa_excluded",
}

# Protected attributes per dataset, IN ORDER (defines PA1/PA2/PA3 slots).
# Names must match the "name" field used in the ground-truth / extraction
# JSON files exactly.
PROTECTED_ATTRS = {
    "credit": ["age", "sex", "foreign_worker"],
    "saudi": ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
    "law": ["gender", "race"],
}

PROBABILITY_TOLERANCE = 0.01


# ============================================================================
# HELPERS
# ============================================================================

def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def clean_value(v):
    """Normalize a raw JSON value: 'NaN' string / None -> None (missing)."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("nan", ""):
        return None
    return v


def values_match(a, b):
    """1/0 match between two feature values, numeric-first with string fallback."""
    try:
        return int(float(a) == float(b))
    except (ValueError, TypeError):
        return int(str(a).strip() == str(b).strip())


def build_name_lookup(record):
    """{feature_name: value} from both most_important_features and features."""
    lookup = {}
    for feat in record.get("most_important_features", []):
        lookup[feat["name"]] = clean_value(feat.get("value"))
    for feat in record.get("features", []):
        lookup[feat["name"]] = clean_value(feat.get("value"))
    return lookup


def discover_other_features(dataset, pa_names):
    """Union of all non-PA feature names appearing anywhere (as a top-SHAP
    feature in some instance, or as a regular feature in another) across
    every ground-truth instance file for this dataset."""
    names = set()
    gt_dir = GT_DIR / dataset
    for path in sorted(gt_dir.glob("instance_*.json")):
        gt = load_json(path)
        if gt is None:
            continue
        for feat in gt.get("most_important_features", []):
            names.add(feat["name"])
        for feat in gt.get("features", []):
            names.add(feat["name"])
    names -= set(pa_names)
    return sorted(names)


def list_instance_ids(dataset):
    gt_dir = GT_DIR / dataset
    ids = []
    for path in sorted(gt_dir.glob("instance_*.json")):
        ids.append(int(path.stem.split("_")[1]))
    return sorted(ids)


# ============================================================================
# PER-ROW COMPUTATION
# ============================================================================

def build_row(dataset, provider, condition_dir, instance_id, pa_names, other_features):
    gt = load_json(GT_DIR / dataset / f"instance_{instance_id}.json")
    ext_path = (EXTRACTIONS_DIR / dataset / condition_dir / provider / EXTRACTOR /
                f"instance_{instance_id}.json")
    ext = load_json(ext_path)

    row = {
        "dataset": dataset,
        "provider": provider,
        "condition": CONDITIONS[condition_dir],
        "instance_id": instance_id,
    }

    if gt is None:
        raise FileNotFoundError(f"Missing ground truth: dataset={dataset} instance={instance_id}")

    gt_lookup = build_name_lookup(gt)
    gt_by_rank = {f["rank"]: f for f in gt.get("most_important_features", [])}
    gt_top3_names = {f["name"] for f in gt.get("most_important_features", [])}

    # <pa_name>_GT columns: the applicant's true value for this attribute,
    # confirmed identical between the ground-truth JSON and <dataset>_adverse.csv.
    for pa in pa_names:
        row[f"{pa}_GT"] = gt_lookup.get(pa) if gt_lookup.get(pa) is not None else np.nan

    if ext is None:
        # Extraction missing entirely: all "_extracted"/"_scoring"/"_mentioned"
        # fields for this row are NaN; GT-side fields are still filled in.
        print(f"WARNING: missing extraction file, filled with NaN: "
              f"{dataset}/{condition_dir}/{provider}/{EXTRACTOR}/instance_{instance_id}.json")
        ext = {}

    ext_by_rank = {f["rank"]: f for f in ext.get("most_important_features", [])}
    ext_top3_by_name = {f["name"]: f for f in ext.get("most_important_features", [])}
    ext_features_by_name = {f["name"]: f for f in ext.get("features", [])}

    # ---- predicted_probability ------------------------------------------
    gt_pp = clean_value(gt.get("predicted_probability"))
    ext_pp = clean_value(ext.get("predicted_probability"))
    row["predicted_probability_GT"] = gt_pp
    row["predicted_probability_extracted"] = ext_pp if ext_pp is not None else np.nan
    if ext_pp is None or gt_pp is None:
        row["predicted_probability_scoring"] = np.nan
    else:
        row["predicted_probability_scoring"] = int(
            np.isclose(float(ext_pp), float(gt_pp), atol=PROBABILITY_TOLERANCE)
        )

    # ---- rank_N / rank_N_sign / rank_N_value ------------------------------
    for n in (1, 2, 3):
        gt_feat = gt_by_rank.get(n)
        gt_name = gt_feat["name"] if gt_feat else None
        gt_sign = gt_feat.get("sign") if gt_feat else None

        ext_feat_at_n = ext_by_rank.get(n)
        ext_name_at_n = ext_feat_at_n["name"] if ext_feat_at_n else None

        row[f"rank_{n}_GT"] = gt_name
        row[f"rank_{n}_extracted"] = ext_name_at_n if ext_name_at_n is not None else np.nan
        row[f"rank_{n}_scoring"] = (
            np.nan if ext_name_at_n is None else int(ext_name_at_n == gt_name)
        )

        # sign: matched by GT rank-N feature's NAME, searched anywhere in
        # the extraction's own top-3 list (see decision 5).
        ext_match_for_sign = ext_top3_by_name.get(gt_name)
        row[f"rank_{n}_sign_GT"] = gt_sign
        if ext_match_for_sign is None:
            row[f"rank_{n}_sign_extracted"] = np.nan
            row[f"rank_{n}_sign_scoring"] = np.nan
        else:
            ext_sign = clean_value(ext_match_for_sign.get("sign"))
            row[f"rank_{n}_sign_extracted"] = ext_sign if ext_sign is not None else np.nan
            if ext_sign is None:
                row[f"rank_{n}_sign_scoring"] = np.nan
            else:
                try:
                    row[f"rank_{n}_sign_scoring"] = int(int(ext_sign) == int(gt_sign))
                except (ValueError, TypeError):
                    row[f"rank_{n}_sign_scoring"] = np.nan

        # value: anchored to GT's rank-N feature NAME (same convention as
        # sign), searched across the ENTIRE extraction output — both its
        # top-3 list and its regular "features" list — so a value stated
        # for this feature counts even if the narrative didn't rank it in
        # its top 3, or ranked it at a different position. This means
        # rank_N_value_scoring can be correct even when rank_N_scoring=0.
        gt_val_for_gt_name = gt_lookup.get(gt_name)
        row[f"rank_{n}_value_GT"] = (
            gt_val_for_gt_name if gt_val_for_gt_name is not None else np.nan
        )

        ext_val_for_gt_name = None
        if gt_name in ext_top3_by_name:
            # extraction ranked this feature somewhere in its top 3 -> always
            # counts as "mentioned", value taken directly from that entry.
            ext_val_for_gt_name = clean_value(ext_top3_by_name[gt_name].get("value"))
        elif gt_name in ext_features_by_name:
            ext_feat_match = ext_features_by_name[gt_name]
            if ext_feat_match.get("mentioned") == 1:
                ext_val_for_gt_name = clean_value(ext_feat_match.get("value"))

        value_mentioned = 1 if ext_val_for_gt_name is not None else 0
        row[f"rank_{n}_value_mentioned"] = value_mentioned
        row[f"rank_{n}_value_extracted"] = (
            ext_val_for_gt_name if ext_val_for_gt_name is not None else np.nan
        )
        if value_mentioned == 0 or gt_val_for_gt_name is None:
            row[f"rank_{n}_value_scoring"] = np.nan
        else:
            row[f"rank_{n}_value_scoring"] = values_match(ext_val_for_gt_name, gt_val_for_gt_name)

    # ---- <pa_name>_mentioned / _extracted / _scoring -----------------------
    # "_mentioned": was this PA referenced in the narrative at all (per the
    # extraction's own mentioned flag), regardless of whether a specific
    # value was stated. "_extracted" holds the value if one was given, NaN
    # otherwise — genuine mentions-without-a-stated-value do occur in the
    # source data (e.g. a gendered pronoun with no explicit value token),
    # so "_mentioned"=1 with "_extracted"=NaN is an expected, real pattern,
    # not a bug. The count of "value actually mentioned" can be recovered
    # downstream as notna(<pa>_extracted).
    for pa in pa_names:
        ext_feat = ext_features_by_name.get(pa) or ext_top3_by_name.get(pa)
        mentioned = 1 if (ext_feat and ext_feat.get("mentioned", 1) == 1) else 0
        # ext_top3_by_name entries have no "mentioned" key; treat presence as mentioned.
        if ext_feat is not None and "mentioned" not in ext_feat:
            mentioned = 1

        if mentioned:
            ext_val = clean_value(ext_feat.get("value"))
        else:
            ext_val = None

        row[f"{pa}_mentioned"] = mentioned
        row[f"{pa}_extracted"] = ext_val if ext_val is not None else np.nan

        gt_val = gt_lookup.get(pa)
        if mentioned == 0 or ext_val is None or gt_val is None:
            row[f"{pa}_scoring"] = np.nan
        else:
            row[f"{pa}_scoring"] = values_match(ext_val, gt_val)

    # ---- other_<feature>_* -------------------------------------------------
    gt_features_by_name = {f["name"]: f for f in gt.get("features", [])}
    for feat_name in other_features:
        col_mentioned = f"other_{feat_name}_mentioned"
        col_gt = f"other_{feat_name}_value_GT"
        col_ext = f"other_{feat_name}_value_extracted"
        col_scoring = f"other_{feat_name}_value_scoring"

        if feat_name in gt_top3_names:
            # This feature is one of THIS instance's top-3 SHAP features;
            # it's already fully represented by the rank_N_* columns above.
            row[col_mentioned] = np.nan
            row[col_gt] = np.nan
            row[col_ext] = np.nan
            row[col_scoring] = np.nan
            continue

        gt_feat = gt_features_by_name.get(feat_name)
        gt_val = clean_value(gt_feat.get("value")) if gt_feat else None
        row[col_gt] = gt_val if gt_val is not None else np.nan

        ext_feat = ext_features_by_name.get(feat_name)
        mentioned = 1 if (ext_feat and ext_feat.get("mentioned") == 1) else 0
        row[col_mentioned] = mentioned

        if mentioned:
            ext_val = clean_value(ext_feat.get("value"))
        else:
            ext_val = None
        row[col_ext] = ext_val if ext_val is not None else np.nan

        if mentioned == 0 or ext_val is None or gt_val is None:
            row[col_scoring] = np.nan
        else:
            row[col_scoring] = values_match(ext_val, gt_val)

    return row


# ============================================================================
# MAIN
# ============================================================================

def build_column_order(pa_names, other_features):
    cols = ["dataset", "provider", "condition", "instance_id"]
    cols += [f"{pa}_GT" for pa in pa_names]
    cols += ["predicted_probability_GT", "predicted_probability_extracted",
             "predicted_probability_scoring"]

    for n in (1, 2, 3):
        cols += [f"rank_{n}_GT", f"rank_{n}_extracted", f"rank_{n}_scoring"]
    for n in (1, 2, 3):
        cols += [f"rank_{n}_sign_GT", f"rank_{n}_sign_extracted", f"rank_{n}_sign_scoring"]
    for n in (1, 2, 3):
        cols += [f"rank_{n}_value_mentioned", f"rank_{n}_value_GT",
                  f"rank_{n}_value_extracted", f"rank_{n}_value_scoring"]

    for pa in pa_names:
        cols += [f"{pa}_mentioned", f"{pa}_extracted", f"{pa}_scoring"]

    for feat in other_features:
        cols += [f"other_{feat}_mentioned", f"other_{feat}_value_GT",
                  f"other_{feat}_value_extracted", f"other_{feat}_value_scoring"]

    return cols


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n{'=' * 80}\n{dataset.upper()}\n{'=' * 80}")

        pa_names = PROTECTED_ATTRS[dataset]
        other_features = discover_other_features(dataset, pa_names)
        instance_ids = list_instance_ids(dataset)
        print(f"  {len(instance_ids)} instances, PAs={pa_names}, "
              f"{len(other_features)} other features")

        rows = []
        for condition_dir in CONDITIONS:
            for provider in NARRATIVE_PROVIDERS:
                for instance_id in instance_ids:
                    row = build_row(dataset, provider, condition_dir, instance_id,
                                     pa_names, other_features)
                    rows.append(row)

        col_order = build_column_order(pa_names, other_features)
        df = pd.DataFrame(rows)
        df = df[col_order]

        out_path = OUTPUT_DIR / f"per_narrative_metrics_{dataset}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Wrote {len(df)} rows, {len(df.columns)} columns -> "
              f"{out_path.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()