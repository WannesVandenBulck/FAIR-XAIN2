#!/usr/bin/env python
"""
Statistical Disparity Analysis for XAI Narrative Fairness (FAIR-XAIN2)
========================================================================

Reads results/fairness_eval/per_narrative_metrics_<dataset>.csv (one row per
narrative = dataset x provider x condition x instance) and runs a full grid of
statistical disparity tests on every feature-coverage and faithfulness metric.

INPUT
-----
results/fairness_eval/per_narrative_metrics_{credit,saudi,student,law}.csv
Each row = one narrative. Columns include:
  - metadata: dataset, provider, condition (include_pa/exclude_pa), instance_id
  - protected attributes (ground truth, constant across provider/condition for
    a given instance_id): pa_<attr> (e.g. pa_age, pa_sex, pa_gender, pa_race...)
  - faithfulness metrics: sign_rank_*_accuracy, rank_*_accuracy,
    predicted_probability_accuracy, *_value_accuracy, total_value_accuracy
  - feature coverage metrics: shap_feature*_mentioned, pa_*_mentioned,
    other_*_mentioned, predicted_probability_mentioned, shap_features_mentioned

COMPARISON FAMILIES
--------------------
A. Total disparity        - group differences WITHIN include_pa (PA disclosed)
B. Indirect disparity      - group differences WITHIN exclude_pa (PA withheld)
C. Disclosure disparity    - include_pa vs exclude_pa, same group   (paired)
D. Provider main effect    - grok vs openai vs deepseek, same instances (paired)
E. Dataset main effect     - credit vs saudi vs student vs law, on metrics
                              that are directly comparable across datasets
                              (dataset-agnostic aggregate/proportion metrics)

Each family is run at multiple pooling levels (see POOLING LEVELS below).
Every test reports an effect size alongside the p-value. Multiple-comparison
correction (Benjamini-Hochberg FDR, and Bonferroni) is applied WITHIN each
(family, pooling_level) group, not globally, per the project's methodology.

POOLING LEVELS
---------------
A, B : per_provider        (dataset x provider, most granular / primary)
       pooled_providers     (dataset, all providers combined - exploratory)
C    : per_provider         (paired disclosed vs withheld, within provider)
D    : overall              (all instances, dataset x condition)
       by_group             (stratified by each protected-attribute group,
                              dataset x condition x attribute x group)
E    : dataset_comparison   (provider x condition, across all 4 datasets)

Two-group PA attributes -> Mann-Whitney U / Fisher's-exact / Wilcoxon / McNemar
Multi-group (>2) PA attributes or providers -> Kruskal-Wallis / Chi-square /
    Friedman / Cochran's Q, with pairwise BH-corrected post-hoc tests.

OUTPUT
------
results/fairness_eval/statistical_disparity_main.csv     (all omnibus/primary tests)
results/fairness_eval/statistical_disparity_posthoc.csv  (pairwise post-hoc tests)
results/fairness_eval/statistical_disparity_summary.txt  (significant findings)

USAGE
-----
python scripts/fairness_eval/statistical_disparity_analysis.py
"""

import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent.parent
FAIRNESS_EVAL_DIR = ROOT / "results" / "fairness_eval"

DATASETS = ["credit", "saudi", "student", "law"]
PROVIDERS = ["grok", "openai", "deepseek"]
CONDITIONS = ["include_pa", "exclude_pa"]

MIN_GROUP_N = 5           # minimum n per group/cell to attempt a test
ALPHA = 0.05

# Protected attributes per dataset: raw column name in the CSV -> config.
# "numeric": True  -> continuous attribute, split into two groups at the
#                      median (computed per unique instance).
# "numeric": False -> already-categorical/coded attribute, "map" gives the
#                      human-readable label for each raw code (falls back to
#                      the raw code as a string if a code isn't in the map).
PROTECTED_ATTRS = {
    "credit": {
        "pa_age": {"numeric": True, "label": "age"},
        "pa_sex": {"numeric": False, "label": "sex",
                   "map": {0: "male", 1: "female"}},
        "pa_foreign_worker": {"numeric": False, "label": "foreign_worker",
                               "map": {1: "yes", 2: "no"}},
    },
    "saudi": {
        "pa_Gender": {"numeric": False, "label": "gender",
                      "map": {0: "Female", 1: "Male"}},
        "pa_Age": {"numeric": True, "label": "age"},
        "pa_Health_Issues": {"numeric": False, "label": "health_issues",
                              "map": {0: "no", 1: "yes"}},
    },
    "student": {
        "pa_sex": {"numeric": False, "label": "sex",
                   "map": {0: "Female", 1: "Male"}},
        "pa_age": {"numeric": True, "label": "age"},
        "pa_health": {"numeric": False, "label": "health",
                      "map": {1: "very bad", 2: "bad", 3: "fair",
                              4: "good", 5: "very good"}},
    },
    "law": {
        "pa_gender": {"numeric": False, "label": "gender",
                      "map": {0: "female", 1: "male"}},
        "pa_race": {"numeric": False, "label": "race",
                    "map": {0: "white", 1: "black", 2: "hispanic",
                            3: "asian", 4: "native american"}},
    },
}

OUTPUT_MAIN = FAIRNESS_EVAL_DIR / "statistical_disparity_main.csv"
OUTPUT_POSTHOC = FAIRNESS_EVAL_DIR / "statistical_disparity_posthoc.csv"
OUTPUT_SUMMARY = FAIRNESS_EVAL_DIR / "statistical_disparity_summary.txt"


# ============================================================================
# DATA LOADING & PREP
# ============================================================================

def load_dataset(dataset):
    """Load a per_narrative_metrics CSV and attach derived aggregate metrics
    and readable group labels for each protected attribute."""
    path = FAIRNESS_EVAL_DIR / f"per_narrative_metrics_{dataset}.csv"
    df = pd.read_csv(path)

    raw_pa_cols = list(PROTECTED_ATTRS[dataset].keys())

    other_mentioned_cols = [c for c in df.columns
                             if c.startswith("other_") and c.endswith("_mentioned")
                             and "_value_" not in c]
    pa_mentioned_cols = [c for c in df.columns
                          if c.startswith("pa_") and c.endswith("_mentioned")
                          and "_value_" not in c]

    # Derived, dataset-agnostic coverage metrics (counts + proportions so
    # they are comparable across datasets with different feature counts).
    df["n_other_features_mentioned"] = df[other_mentioned_cols].sum(axis=1)
    df["prop_other_features_mentioned"] = df["n_other_features_mentioned"] / max(len(other_mentioned_cols), 1)
    df["n_pa_mentioned"] = df[pa_mentioned_cols].sum(axis=1)
    df["prop_pa_mentioned"] = df["n_pa_mentioned"] / max(len(pa_mentioned_cols), 1)

    # Readable group label per protected attribute.
    group_cols = []
    for raw_col, cfg in PROTECTED_ATTRS[dataset].items():
        group_col = f"group__{cfg['label']}"
        if cfg["numeric"]:
            # median computed on unique instances so repeated rows don't skew it
            median_val = df.drop_duplicates("instance_id")[raw_col].median()
            df[group_col] = np.where(df[raw_col] < median_val,
                                      f"<{median_val:g}", f">={median_val:g}")
        else:
            value_map = cfg.get("map", {})
            df[group_col] = df[raw_col].map(lambda v: value_map.get(v, str(v)))
        group_cols.append((cfg["label"], group_col))

    metadata_cols = {"dataset", "provider", "condition", "instance_id",
                      "predicted_probability"}
    excluded = metadata_cols | set(raw_pa_cols) | {g for _, g in group_cols}
    metric_cols = [c for c in df.columns if c not in excluded]

    return df, group_cols, metric_cols


def classify_metric(series):
    """binary (0/1) vs continuous, based on observed non-null values."""
    vals = series.dropna().unique()
    if len(vals) == 0:
        return None
    if set(np.round(vals.astype(float), 6)).issubset({0.0, 1.0}):
        return "binary"
    return "continuous"


# ============================================================================
# CORE STATISTICAL TESTS
# ============================================================================

def effect_size_rank_biserial_unpaired(x, y):
    n1, n2 = len(x), len(y)
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return (2 * u_stat) / (n1 * n2) - 1


def effect_size_rank_biserial_paired(diffs):
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return np.nan
    n_pos = (diffs > 0).sum()
    n_neg = (diffs < 0).sum()
    return (n_pos - n_neg) / (n_pos + n_neg)


def cramers_v(chi2, contingency_table):
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    if n == 0 or k == 0:
        return np.nan
    return np.sqrt((chi2 / n) / k)


def odds_ratio_2x2(table):
    a, b = table[0]
    c, d = table[1]
    # Haldane-Anscombe correction if any cell is zero
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a * d) / (b * c)


def test_unpaired_2group(vals1, vals2, kind, name1, name2):
    """Two independent groups. kind in {'binary','continuous'}."""
    n1, n2 = len(vals1), len(vals2)
    if n1 < MIN_GROUP_N or n2 < MIN_GROUP_N:
        return None

    if kind == "binary":
        s1, f1 = int(np.nansum(vals1)), n1 - int(np.nansum(vals1))
        s2, f2 = int(np.nansum(vals2)), n2 - int(np.nansum(vals2))
        table = np.array([[s1, f1], [s2, f2]])
        if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            return None
        expected = stats.contingency.expected_freq(table)
        if (expected < 5).any():
            _, p = stats.fisher_exact(table)
            test_name = "Fisher exact"
            stat = np.nan
        else:
            stat, p, _, _ = stats.chi2_contingency(table, correction=False)
            test_name = "Chi-square"
        try:
            eff = odds_ratio_2x2(table)
        except ZeroDivisionError:
            eff = np.nan
        return dict(test=test_name, statistic=stat, p_value=p,
                    effect_size=eff, effect_size_type="odds_ratio",
                    n1=n1, n2=n2, mean1=np.nanmean(vals1), mean2=np.nanmean(vals2),
                    group1=name1, group2=name2)

    else:  # continuous
        vals1c, vals2c = vals1[~np.isnan(vals1)], vals2[~np.isnan(vals2)]
        if len(vals1c) < MIN_GROUP_N or len(vals2c) < MIN_GROUP_N:
            return None
        try:
            stat, p = stats.mannwhitneyu(vals1c, vals2c, alternative="two-sided")
            eff = effect_size_rank_biserial_unpaired(vals1c, vals2c)
        except ValueError:
            return None
        return dict(test="Mann-Whitney U", statistic=stat, p_value=p,
                    effect_size=eff, effect_size_type="rank_biserial_r",
                    n1=len(vals1c), n2=len(vals2c),
                    mean1=np.mean(vals1c), mean2=np.mean(vals2c),
                    group1=name1, group2=name2)


def test_unpaired_kgroup(groups_dict, kind):
    """groups_dict: {group_name: np.array of values}. kind in {'binary','continuous'}."""
    names = list(groups_dict.keys())
    arrays = [groups_dict[n] for n in names]
    sizes = [len(a) for a in arrays]
    if any(s < MIN_GROUP_N for s in sizes) or len(names) < 3:
        return None, []

    posthoc = []
    if kind == "binary":
        table = np.array([[int(np.nansum(a)), len(a) - int(np.nansum(a))] for a in arrays])
        # degenerate table (a whole row/column is all-zero) -> chi-square undefined
        if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            return None, []
        expected = stats.contingency.expected_freq(table)
        stat, p, _, _ = stats.chi2_contingency(table, correction=False)
        eff = cramers_v(stat, table)
        omnibus = dict(test="Chi-square", statistic=stat, p_value=p,
                        effect_size=eff, effect_size_type="cramers_v",
                        n_groups=len(names), n_total=table.sum(),
                        small_expected_counts=bool((expected < 5).any()))
        for (n1, n2) in combinations(names, 2):
            r = test_unpaired_2group(groups_dict[n1], groups_dict[n2], "binary", n1, n2)
            if r:
                posthoc.append(r)
    else:
        clean = [a[~np.isnan(a)] for a in arrays]
        if any(len(a) < MIN_GROUP_N for a in clean):
            return None, []
        stat, p = stats.kruskal(*clean)
        n_total = sum(len(a) for a in clean)
        k = len(clean)
        eps2 = (stat - k + 1) / (n_total - k) if n_total > k else np.nan
        omnibus = dict(test="Kruskal-Wallis", statistic=stat, p_value=p,
                        effect_size=eps2, effect_size_type="epsilon_squared",
                        n_groups=k, n_total=n_total, small_expected_counts=False)
        for (n1, n2) in combinations(names, 2):
            r = test_unpaired_2group(groups_dict[n1], groups_dict[n2], "continuous", n1, n2)
            if r:
                posthoc.append(r)

    return omnibus, posthoc


def test_paired_2cond(vals_a, vals_b, kind):
    """Paired samples (same instances), 2 conditions. Arrays must be aligned."""
    mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
    a, b = vals_a[mask], vals_b[mask]
    n = len(a)
    if n < MIN_GROUP_N:
        return None

    if kind == "binary":
        table = np.zeros((2, 2))
        for av, bv in zip(a, b):
            table[int(av), int(bv)] += 1
        if table[0, 1] + table[1, 0] == 0:
            return None  # no discordant pairs, McNemar undefined
        result = mcnemar(table, exact=(table[0, 1] + table[1, 0] < 25), correction=True)
        b_disc, c_disc = table[0, 1], table[1, 0]
        eff = (c_disc + 0.5) / (b_disc + 0.5)  # odds-ratio-like, Haldane corrected
        return dict(test="McNemar", statistic=result.statistic, p_value=result.pvalue,
                    effect_size=eff, effect_size_type="discordant_ratio",
                    n=n, mean_a=np.mean(a), mean_b=np.mean(b))
    else:
        diffs = a - b
        if np.all(diffs == 0):
            return None
        try:
            stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            return None
        eff = effect_size_rank_biserial_paired(diffs)
        return dict(test="Wilcoxon signed-rank", statistic=stat, p_value=p,
                    effect_size=eff, effect_size_type="matched_rank_biserial_r",
                    n=n, mean_a=np.mean(a), mean_b=np.mean(b))


def test_paired_kgroup(wide_df, kind):
    """wide_df: rows = instances, columns = providers, complete cases only."""
    wide_df = wide_df.dropna()
    n = len(wide_df)
    k = wide_df.shape[1]
    if n < MIN_GROUP_N or k < 3:
        return None, []

    posthoc = []
    if kind == "binary":
        result = cochrans_q(wide_df.values)
        omnibus = dict(test="Cochran's Q", statistic=result.statistic, p_value=result.pvalue,
                        effect_size=np.nan, effect_size_type=None,
                        n=n, k=k)
        for c1, c2 in combinations(wide_df.columns, 2):
            r = test_paired_2cond(wide_df[c1].values, wide_df[c2].values, "binary")
            if r:
                r["group1"], r["group2"] = c1, c2
                posthoc.append(r)
    else:
        stat, p = stats.friedmanchisquare(*[wide_df[c] for c in wide_df.columns])
        kendalls_w = stat / (n * (k - 1)) if n * (k - 1) > 0 else np.nan
        omnibus = dict(test="Friedman", statistic=stat, p_value=p,
                        effect_size=kendalls_w, effect_size_type="kendalls_w",
                        n=n, k=k)
        for c1, c2 in combinations(wide_df.columns, 2):
            r = test_paired_2cond(wide_df[c1].values, wide_df[c2].values, "continuous")
            if r:
                r["group1"], r["group2"] = c1, c2
                posthoc.append(r)

    return omnibus, posthoc


# ============================================================================
# FAMILY DRIVERS
# ============================================================================

main_rows = []
posthoc_rows = []


def record_main(row):
    main_rows.append(row)


def record_posthoc(row):
    posthoc_rows.append(row)


def family_AB(dataset, df, group_cols, metric_cols, condition, family_label):
    """A (include_pa) / B (exclude_pa): compare PA groups within one condition."""
    sub = df[df["condition"] == condition]

    for attr_label, group_col in group_cols:
        # --- per_provider level ---
        for provider in PROVIDERS:
            psub = sub[sub["provider"] == provider]
            for metric in metric_cols:
                kind = classify_metric(psub[metric])
                if kind is None:
                    continue
                groups = {g: psub.loc[psub[group_col] == g, metric].to_numpy(dtype=float)
                          for g in psub[group_col].dropna().unique()}
                groups = {g: v for g, v in groups.items() if len(v) >= MIN_GROUP_N}
                base = dict(family=family_label, dataset=dataset, pooling_level="per_provider",
                            provider=provider, condition=condition, attribute=attr_label, metric=metric,
                            metric_type=kind)
                if len(groups) == 2:
                    (n1, v1), (n2, v2) = groups.items()
                    r = test_unpaired_2group(v1, v2, kind, n1, n2)
                    if r:
                        record_main({**base, **r})
                elif len(groups) > 2:
                    omni, post = test_unpaired_kgroup(groups, kind)
                    if omni:
                        record_main({**base, **omni})
                        for p in post:
                            record_posthoc({**base, **p})

        # --- pooled_providers level (exploratory) ---
        for metric in metric_cols:
            kind = classify_metric(sub[metric])
            if kind is None:
                continue
            groups = {g: sub.loc[sub[group_col] == g, metric].to_numpy(dtype=float)
                      for g in sub[group_col].dropna().unique()}
            groups = {g: v for g, v in groups.items() if len(v) >= MIN_GROUP_N}
            base = dict(family=family_label, dataset=dataset, pooling_level="pooled_providers",
                        provider="pooled", condition=condition, attribute=attr_label, metric=metric,
                        metric_type=kind)
            if len(groups) == 2:
                (n1, v1), (n2, v2) = groups.items()
                r = test_unpaired_2group(v1, v2, kind, n1, n2)
                if r:
                    record_main({**base, **r})
            elif len(groups) > 2:
                omni, post = test_unpaired_kgroup(groups, kind)
                if omni:
                    record_main({**base, **omni})
                    for p in post:
                        record_posthoc({**base, **p})


def family_C(dataset, df, group_cols, metric_cols):
    """C: disclosure disparity - include_pa vs exclude_pa, paired by instance,
    within each group level, per provider."""
    for attr_label, group_col in group_cols:
        for provider in PROVIDERS:
            psub = df[df["provider"] == provider]
            for group_val in psub[group_col].dropna().unique():
                gsub = psub[psub[group_col] == group_val]
                if ("include_pa" not in gsub["condition"].values or
                        "exclude_pa" not in gsub["condition"].values):
                    continue
                wide = gsub.pivot_table(index="instance_id", columns="condition",
                                         values=metric_cols)
                for metric in metric_cols:
                    try:
                        a = wide[(metric, "include_pa")]
                        b = wide[(metric, "exclude_pa")]
                    except KeyError:
                        continue
                    kind = classify_metric(pd.concat([a, b]))
                    if kind is None:
                        continue
                    r = test_paired_2cond(a.to_numpy(dtype=float), b.to_numpy(dtype=float), kind)
                    if r:
                        record_main(dict(family="C_disclosure_disparity", dataset=dataset,
                                          pooling_level="per_provider", provider=provider,
                                          condition="include_pa_vs_exclude_pa",
                                          attribute=attr_label, group=group_val,
                                          metric=metric, metric_type=kind, **r))


def family_D(dataset, df, group_cols, metric_cols):
    """D: provider main effect - grok vs openai vs deepseek, paired by instance."""
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]

        # --- overall (all instances) ---
        for metric in metric_cols:
            kind = classify_metric(sub[metric])
            if kind is None:
                continue
            wide = sub.pivot_table(index="instance_id", columns="provider", values=metric)
            wide = wide[[p for p in PROVIDERS if p in wide.columns]]
            omni, post = test_paired_kgroup(wide, kind)
            base = dict(family="D_provider_main_effect", dataset=dataset,
                        pooling_level="overall", provider="grok_vs_openai_vs_deepseek",
                        condition=condition, attribute="n/a", metric=metric, metric_type=kind)
            if omni:
                record_main({**base, **omni})
                for p in post:
                    record_posthoc({**base, **p})

        # --- by_group (stratified) ---
        for attr_label, group_col in group_cols:
            for group_val in sub[group_col].dropna().unique():
                gsub = sub[sub[group_col] == group_val]
                for metric in metric_cols:
                    kind = classify_metric(gsub[metric])
                    if kind is None:
                        continue
                    wide = gsub.pivot_table(index="instance_id", columns="provider", values=metric)
                    wide = wide[[p for p in PROVIDERS if p in wide.columns]]
                    omni, post = test_paired_kgroup(wide, kind)
                    base = dict(family="D_provider_main_effect", dataset=dataset,
                                pooling_level="by_group", provider="grok_vs_openai_vs_deepseek",
                                condition=condition, attribute=attr_label, group=group_val,
                                metric=metric, metric_type=kind)
                    if omni:
                        record_main({**base, **omni})
                        for p in post:
                            record_posthoc({**base, **p})


DATASET_AGNOSTIC_METRICS = [
    "sign_total_accuracy", "rank_total_accuracy", "total_value_accuracy",
    "predicted_probability_accuracy", "predicted_probability_mentioned",
    "shap_features_mentioned", "n_pa_mentioned", "prop_pa_mentioned",
    "n_other_features_mentioned", "prop_other_features_mentioned",
]


def family_E(all_dfs):
    """E: dataset main effect on dataset-agnostic metrics, unpaired across
    the 4 datasets (different instances entirely -> no pairing possible)."""
    for provider in PROVIDERS:
        for condition in CONDITIONS:
            for metric in DATASET_AGNOSTIC_METRICS:
                groups = {}
                for dataset, df in all_dfs.items():
                    sub = df[(df["provider"] == provider) & (df["condition"] == condition)]
                    if metric not in sub.columns:
                        continue
                    vals = sub[metric].to_numpy(dtype=float)
                    if len(vals) >= MIN_GROUP_N:
                        groups[dataset] = vals
                if len(groups) < 2:
                    continue
                kind = classify_metric(pd.Series(np.concatenate(list(groups.values()))))
                if kind is None:
                    continue
                base = dict(family="E_dataset_main_effect", dataset="credit_vs_saudi_vs_student_vs_law",
                            pooling_level="dataset_comparison", provider=provider, condition=condition,
                            attribute="n/a", metric=metric, metric_type=kind)
                if len(groups) == 2:
                    (n1, v1), (n2, v2) = groups.items()
                    r = test_unpaired_2group(v1, v2, kind, n1, n2)
                    if r:
                        record_main({**base, **r})
                else:
                    omni, post = test_unpaired_kgroup(groups, kind)
                    if omni:
                        record_main({**base, **omni})
                        for p in post:
                            record_posthoc({**base, **p})


# ============================================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================================

def apply_correction(df):
    """BH-FDR and Bonferroni correction, applied within each
    (family, pooling_level) group, not globally."""
    if df.empty:
        return df
    df = df.copy()
    df["p_value_fdr"] = np.nan
    df["p_value_bonferroni"] = np.nan
    for (family, level), idx in df.groupby(["family", "pooling_level"]).groups.items():
        pvals = df.loc[idx, "p_value"].values
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            continue
        _, fdr, _, _ = multipletests(pvals[valid], alpha=ALPHA, method="fdr_bh")
        _, bonf, _, _ = multipletests(pvals[valid], alpha=ALPHA, method="bonferroni")
        valid_idx = np.array(idx)[valid]
        df.loc[valid_idx, "p_value_fdr"] = fdr
        df.loc[valid_idx, "p_value_bonferroni"] = bonf
    df["significant_raw"] = df["p_value"] < ALPHA
    df["significant_fdr"] = df["p_value_fdr"] < ALPHA
    df["significant_bonferroni"] = df["p_value_bonferroni"] < ALPHA
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("STATISTICAL DISPARITY ANALYSIS")
    print("=" * 100)

    all_dfs = {}
    all_group_cols = {}
    all_metric_cols = {}

    for dataset in DATASETS:
        print(f"\nLoading {dataset}...")
        df, group_cols, metric_cols = load_dataset(dataset)
        all_dfs[dataset] = df
        all_group_cols[dataset] = group_cols
        all_metric_cols[dataset] = metric_cols
        print(f"  {len(df)} narratives, {len(metric_cols)} metrics, "
              f"attributes: {[g for g, _ in group_cols]}")

    for dataset in DATASETS:
        df = all_dfs[dataset]
        group_cols = all_group_cols[dataset]
        metric_cols = all_metric_cols[dataset]

        print(f"\nRunning tests for {dataset.upper()}...")

        print("  Family A (total disparity)...")
        family_AB(dataset, df, group_cols, metric_cols, "include_pa", "A_total_disparity")

        print("  Family B (indirect disparity)...")
        family_AB(dataset, df, group_cols, metric_cols, "exclude_pa", "B_indirect_disparity")

        print("  Family C (disclosure disparity)...")
        family_C(dataset, df, group_cols, metric_cols)

        print("  Family D (provider main effect)...")
        family_D(dataset, df, group_cols, metric_cols)

    print("\nRunning Family E (dataset main effect)...")
    family_E(all_dfs)

    df_main = pd.DataFrame(main_rows)
    df_posthoc = pd.DataFrame(posthoc_rows)

    print(f"\nTotal primary/omnibus tests: {len(df_main)}")
    print(f"Total post-hoc pairwise tests: {len(df_posthoc)}")

    print("\nApplying multiple-comparison correction (within family x pooling_level)...")
    df_main = apply_correction(df_main)
    df_posthoc = apply_correction(df_posthoc)

    FAIRNESS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df_main.to_csv(OUTPUT_MAIN, index=False)
    df_posthoc.to_csv(OUTPUT_POSTHOC, index=False)
    print(f"Saved: {OUTPUT_MAIN.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_POSTHOC.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Summary of significant findings
    # ------------------------------------------------------------------
    lines = ["=" * 100, "SIGNIFICANT FINDINGS SUMMARY", "=" * 100, ""]
    for label, source in [("PRIMARY/OMNIBUS TESTS", df_main), ("POST-HOC PAIRWISE TESTS", df_posthoc)]:
        lines.append(f"\n{'#' * 100}\n# {label}\n{'#' * 100}")
        for corr_col, corr_label in [("significant_raw", "RAW (uncorrected)"),
                                      ("significant_fdr", "FDR-corrected"),
                                      ("significant_bonferroni", "Bonferroni-corrected")]:
            if source.empty or corr_col not in source.columns:
                continue
            sig = source[source[corr_col] == True]  # noqa: E712
            lines.append(f"\n{corr_label}: {len(sig)} significant findings")
            lines.append("-" * 100)
            for _, row in sig.iterrows():
                p_report = row.get({"significant_raw": "p_value",
                                     "significant_fdr": "p_value_fdr",
                                     "significant_bonferroni": "p_value_bonferroni"}[corr_col], np.nan)
                eff = row.get("effect_size", np.nan)
                eff_type = row.get("effect_size_type", "")
                extra = f" attr={row.get('attribute','')}" if pd.notna(row.get('attribute', np.nan)) else ""
                lines.append(
                    f"{row['family']:26} | {row['dataset']:12} | {row['pooling_level']:16} | "
                    f"{row.get('provider',''):28} | {row.get('condition',''):26} | "
                    f"{row['metric']:35} |{extra} | p={p_report:.4g} | "
                    f"{eff_type}={eff:.3f}" if pd.notna(eff) else
                    f"{row['family']:26} | {row['dataset']:12} | {row['pooling_level']:16} | "
                    f"{row.get('provider',''):28} | {row.get('condition',''):26} | "
                    f"{row['metric']:35} |{extra} | p={p_report:.4g}"
                )

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {OUTPUT_SUMMARY.relative_to(ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()