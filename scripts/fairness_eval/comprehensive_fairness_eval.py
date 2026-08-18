#!/usr/bin/env python
"""
Comprehensive Fairness Evaluation Script

Computes per-narrative faithfulness and feature coverage metrics, then runs
multiple comparison families with appropriate statistical tests, effect sizes,
and FDR corrections. Outputs long-format results table for analysis.

Comparison families:
  A. Total disparity (groups within disclosed condition)
  B. Indirect disparity (groups within withheld condition)
  C. Disclosure disparity (disclosed vs withheld, paired by instance)
  D. Provider main effect (grok vs openai vs deepseek, paired)
  E. Dataset main effect (exploratory, dataset-agnostic metrics only)
  F. Interaction checks (group effect × provider)

At three pooling levels:
  1. Per dataset × per provider (primary, granular)
  2. Per dataset, pooled over providers (provider as blocking factor)
  3. Fully pooled (exploratory, flag non-independence)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Statistical libraries
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu, wilcoxon, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = {
    "credit": {
        "num_instances": 97,
        "protected_attrs": ["age", "sex", "foreign_worker"],
        "group_by": {"age": "median", "sex": "value", "foreign_worker": "value"},
        "adverse_csv": "datasets_prep/data/credit_dataset/credit_adverse.csv",
    },
    "law": {
        "num_instances": 308,
        "protected_attrs": ["gender", "race"],
        "group_by": {"gender": "value", "race": "value"},
        "adverse_csv": "datasets_prep/data/law_dataset/law_adverse.csv",
    },
    "saudi": {
        "num_instances": 106,
        "protected_attrs": ["Gender", "Age", "Health_Issues"],
        "group_by": {"Gender": "value", "Age": "median", "Health_Issues": "value"},
        "adverse_csv": "datasets_prep/data/saudi_dataset/saudi_adverse.csv",
    },
    "student": {
        "num_instances": 73,
        "protected_attrs": ["sex", "age", "health"],
        "group_by": {"sex": "value", "age": "median", "health": "value"},
        "adverse_csv": "datasets_prep/data/student_dataset/student_adverse.csv",
    }
}

PROVIDERS = ["grok", "openai", "deepseek"]
CONDITIONS = ["include_pa", "exclude_pa"]  # include_pa = disclosed, exclude_pa = withheld

# Dataset-agnostic metrics for family E
DATASET_AGNOSTIC_METRICS = [
    "overall_faithfulness",
    "rank_total_accuracy",
    "sign_accuracy",
    "shap_features_mentioned",
    "other_features_mentioned",
    "protected_features_mentioned"
]

# ============================================================================
# METRIC COMPUTATION
# ============================================================================

# NOTE: Metrics are now pre-computed in generate_per_narrative_metrics.py
# This script loads pre-computed metrics directly from CSVs and runs statistical
# tests on them. The metric computation functions have been removed.

# ============================================================================
# DATA ORGANIZATION
# ============================================================================

def load_per_narrative_metrics():
    """Load pre-computed per-narrative metrics CSVs (one per dataset)."""
    dfs = []
    for dataset in DATASETS.keys():
        csv_path = f"results/fairness_eval/per_narrative_metrics_{dataset}.csv"
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"Metrics CSV not found at {csv_path}. "
                "Run generate_per_narrative_metrics.py first."
            )
        df = pd.read_csv(csv_path)
        dfs.append(df)
    
    # Concatenate all datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def assign_group_from_value(value, pa_attr, dataset_config):
    """Assign group membership from protected attribute value."""
    group_method = dataset_config["group_by"].get(pa_attr, "value")
    
    if group_method == "median":
        # For numeric PAs, bin by median
        # Would need to compute median from data; for now use "value"
        return str(value)
    else:
        return str(value)

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

class FairnessTestSuite:
    """Suite of statistical tests for fairness comparisons."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, family, pooling_level, dataset, provider, condition, pa_attr,
                   metric, test_name, groups, test_stat, p_value, effect_size, direction):
        """Record a test result."""
        self.results.append({
            "family": family,
            "pooling_level": pooling_level,
            "dataset": dataset,
            "provider": provider,
            "condition": condition,
            "pa_attr": pa_attr,
            "metric": metric,
            "test": test_name,
            "groups": groups,
            "test_statistic": test_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "direction": direction,
        })
    
    def run_family_a(self, df):
        """Total disparity: groups within disclosed (include_pa) condition."""
        print("\n" + "="*80)
        print("FAMILY A: Total Disparity (Groups within Disclosed Condition)")
        print("="*80)
        
        for dataset in DATASETS:
            protected_attrs = DATASETS[dataset]["protected_attrs"]
            
            for provider in PROVIDERS:
                for pa_attr in protected_attrs:
                    pa_col = f"pa_{pa_attr}"
                    if pa_col not in df.columns:
                        continue
                    
                    for metric in df.columns:
                        # Skip non-metric columns
                        if metric.startswith(("dataset", "provider", "condition", "instance", "pa_", "predicted")):
                            continue
                        if df[metric].dtype not in [np.float64, np.int64, float, int]:
                            continue
                        
                        # Filter data
                        subset = df[
                            (df["dataset"] == dataset) &
                            (df["provider"] == provider) &
                            (df["condition"] == "include_pa")
                        ].copy()
                        
                        if subset.empty:
                            continue
                        
                        # Create groups from PA values (optional binning for numeric)
                        subset["group"] = subset[pa_col].astype(str)
                        
                        groups = subset["group"].unique()
                        if len(groups) < 2:
                            continue
                        
                        # Run test
                        self._run_groupwise_test(
                            subset, "group", metric,
                            family="A", pooling=1, dataset=dataset, provider=provider,
                            condition="include_pa", pa_attr=pa_attr
                        )
    
    def run_family_b(self, df):
        """Indirect disparity: groups within withheld (exclude_pa) condition."""
        print("\n" + "="*80)
        print("FAMILY B: Indirect Disparity (Groups within Withheld Condition)")
        print("="*80)
        
        for dataset in DATASETS:
            protected_attrs = DATASETS[dataset]["protected_attrs"]
            
            for provider in PROVIDERS:
                for pa_attr in protected_attrs:
                    pa_col = f"pa_{pa_attr}"
                    if pa_col not in df.columns:
                        continue
                    
                    for metric in df.columns:
                        if metric.startswith(("dataset", "provider", "condition", "instance", "pa_", "predicted")):
                            continue
                        if df[metric].dtype not in [np.float64, np.int64, float, int]:
                            continue
                        
                        subset = df[
                            (df["dataset"] == dataset) &
                            (df["provider"] == provider) &
                            (df["condition"] == "exclude_pa")
                        ].copy()
                        
                        if subset.empty:
                            continue
                        
                        subset["group"] = subset[pa_col].astype(str)
                        groups = subset["group"].unique()
                        if len(groups) < 2:
                            continue
                        
                        self._run_groupwise_test(
                            subset, "group", metric,
                            family="B", pooling=1, dataset=dataset, provider=provider,
                            condition="exclude_pa", pa_attr=pa_attr
                        )
    
    def run_family_c(self, df):
        """Disclosure disparity: disclosed vs withheld, paired by instance."""
        print("\n" + "="*80)
        print("FAMILY C: Disclosure Disparity (Paired: Include vs Exclude PA)")
        print("="*80)
        
        for dataset in DATASETS:
            protected_attrs = DATASETS[dataset]["protected_attrs"]
            
            for provider in PROVIDERS:
                for pa_attr in protected_attrs:
                    pa_col = f"pa_{pa_attr}"
                    if pa_col not in df.columns:
                        continue
                    
                    for metric in df.columns:
                        if metric.startswith(("dataset", "provider", "condition", "instance", "pa_", "predicted")):
                            continue
                        if df[metric].dtype not in [np.float64, np.int64, float, int]:
                            continue
                        
                        # Get instances available in both conditions
                        disclosed = df[
                            (df["dataset"] == dataset) &
                            (df["provider"] == provider) &
                            (df["condition"] == "include_pa")
                        ].copy()
                        
                        withheld = df[
                            (df["dataset"] == dataset) &
                            (df["provider"] == provider) &
                            (df["condition"] == "exclude_pa")
                        ].copy()
                        
                        if disclosed.empty or withheld.empty:
                            continue
                        
                        # Merge on instance_id
                        merged = disclosed.merge(
                            withheld[["instance_id", metric]],
                            on="instance_id",
                            suffixes=("_disclosed", "_withheld")
                        )
                        
                        if len(merged) < 5:
                            continue
                        
                        # Paired test per group (within each PA group)
                        merged["group"] = merged[pa_col].astype(str)
                        
                        for group_val in merged["group"].unique():
                            group_data = merged[merged["group"] == group_val].copy()
                            if len(group_data) < 3:
                                continue
                            
                            # Drop rows where either value is NaN to maintain pairing
                            group_data = group_data.dropna(subset=[f"{metric}_disclosed", f"{metric}_withheld"])
                            if len(group_data) < 3:
                                continue
                            
                            disclosed_vals = group_data[f"{metric}_disclosed"].values
                            withheld_vals = group_data[f"{metric}_withheld"].values
                            
                            # Wilcoxon signed-rank test
                            stat, pval = wilcoxon(disclosed_vals, withheld_vals)
                            
                            # Matched rank-biserial
                            effect = self._matched_rank_biserial(disclosed_vals, withheld_vals)
                            
                            direction = "disclosed > withheld" if stat > 0 else "disclosed < withheld"
                            
                            self.add_result(
                                family="C", pooling_level=1,
                                dataset=dataset, provider=provider, condition="both",
                                pa_attr=pa_attr, metric=metric, test_name="Wilcoxon SR",
                                groups=f"{group_val}", test_stat=stat, p_value=pval,
                                effect_size=effect, direction=direction
                            )
    
    def run_family_d(self, df):
        """Provider main effect: grok vs openai vs deepseek, paired."""
        print("\n" + "="*80)
        print("FAMILY D: Provider Main Effect (Paired across Providers)")
        print("="*80)
        
        for dataset in DATASETS:
            protected_attrs = DATASETS[dataset]["protected_attrs"]
            
            for condition in CONDITIONS:
                for pa_attr in protected_attrs:
                    pa_col = f"pa_{pa_attr}"
                    if pa_col not in df.columns:
                        continue
                    
                    for metric in df.columns:
                        if metric.startswith(("dataset", "provider", "condition", "instance", "pa_", "predicted")):
                            continue
                        if df[metric].dtype not in [np.float64, np.int64, float, int]:
                            continue
                        
                        # Get data per provider
                        provider_data = {}
                        for provider in PROVIDERS:
                            subset = df[
                                (df["dataset"] == dataset) &
                                (df["provider"] == provider) &
                                (df["condition"] == condition)
                            ].copy()
                            if not subset.empty:
                                provider_data[provider] = subset.set_index("instance_id")[metric]
                        
                        if len(provider_data) < 2:
                            continue
                        
                        # Friedman test (paired across 3 providers)
                        # Convert set to list for pandas .loc indexing
                        instances = list(set.intersection(*[set(data.index) for data in provider_data.values()]))
                        if len(instances) < 5:
                            continue
                        
                        arrays = [provider_data[p].loc[instances].values for p in PROVIDERS]
                        stat, pval = stats.friedmanchisquare(*arrays)
                        
                        self.add_result(
                            family="D", pooling_level=1,
                            dataset=dataset, provider="all", condition=condition,
                            pa_attr=pa_attr, metric=metric, test_name="Friedman",
                            groups="grok,openai,deepseek", test_stat=stat, p_value=pval,
                            effect_size=np.nan, direction="varies"
                        )
    
    def _run_groupwise_test(self, df, group_col, metric, family, pooling, dataset, provider, condition, pa_attr):
        """Run appropriate test for groupwise comparison."""
        groups = df[group_col].unique()
        if len(groups) < 2:
            return
        
        if len(groups) == 2:
            # Mann-Whitney U for continuous
            g1_data = df[df[group_col] == groups[0]][metric].dropna()
            g2_data = df[df[group_col] == groups[1]][metric].dropna()
            
            if len(g1_data) < 3 or len(g2_data) < 3:
                return
            
            stat, pval = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
            effect = self._rank_biserial(g1_data, g2_data)
            
            test_name = "Mann-Whitney U"
        else:
            # Kruskal-Wallis for >2 groups
            group_arrays = [df[df[group_col] == g][metric].dropna() for g in groups]
            if any(len(arr) < 3 for arr in group_arrays):
                return
            
            stat, pval = kruskal(*group_arrays)
            effect = self._epsilon_squared(group_arrays)
            test_name = "Kruskal-Wallis"
        
        direction = f"{groups[0]} vs {groups[1] if len(groups) == 2 else f'{len(groups)} groups'}"
        
        self.add_result(
            family=family, pooling_level=pooling,
            dataset=dataset, provider=provider, condition=condition,
            pa_attr=pa_attr, metric=metric, test_name=test_name,
            groups=str(groups), test_stat=stat, p_value=pval,
            effect_size=effect, direction=direction
        )
    
    @staticmethod
    def _rank_biserial(g1, g2):
        """Rank-biserial correlation for Mann-Whitney U."""
        n1, n2 = len(g1), len(g2)
        r = 1 - (2*sum(g1) / (n1 * n2 * (n1 + n2)))
        return r
    
    @staticmethod
    def _matched_rank_biserial(paired1, paired2):
        """Matched rank-biserial for Wilcoxon."""
        diffs = paired1 - paired2
        n = len(diffs)
        r = 1 - (2*np.abs(np.sum(np.sign(diffs))) / n)
        return r
    
    @staticmethod
    def _epsilon_squared(groups):
        """Epsilon-squared for Kruskal-Wallis."""
        all_data = np.concatenate(groups)
        n = len(all_data)
        k = len(groups)
        H = stats.kruskal(*groups)[0]
        eps2 = (H - k + 1) / (n - k)
        return max(0, eps2)
    
    def apply_fdr_correction(self):
        """Apply Benjamini-Hochberg FDR correction per family+pooling level."""
        results_df = pd.DataFrame(self.results)
        
        if results_df.empty:
            return results_df
        
        # Group by family and pooling level
        for (family, pooling), group in results_df.groupby(["family", "pooling_level"]):
            indices = group.index
            p_vals = group["p_value"].values
            
            # Apply BH correction
            reject, corrected_pvals, _, _ = multipletests(p_vals, method="fdr_bh")
            
            results_df.loc[indices, "p_value_fdr"] = corrected_pvals
            results_df.loc[indices, "significant_fdr"] = reject
        
        return results_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE FAIRNESS EVALUATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)
    
    # Step 1: Load per-narrative metrics CSV
    print("\n[1/4] Loading per-narrative metrics...")
    df = load_per_narrative_metrics()
    print(f"✓ Loaded {len(df)} narratives")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Datasets: {df['dataset'].unique()}")
    print(f"  Providers: {df['provider'].unique()}")
    print(f"  Conditions: {df['condition'].unique()}")
    
    # Step 2: Run tests
    print("\n[2/4] Running statistical tests (Families A-D)...")
    suite = FairnessTestSuite()
    suite.run_family_a(df)
    suite.run_family_b(df)
    suite.run_family_c(df)
    suite.run_family_d(df)
    
    # Step 3: Apply corrections
    print("\n[3/4] Applying FDR corrections...")
    results_df = suite.apply_fdr_correction()
    print(f"✓ Completed {len(results_df)} statistical tests")
    
    # Step 4: Export
    print("\n[4/4] Exporting results...")
    output_path = "results/fairness_eval/comprehensive_fairness_eval.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved to {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total tests: {len(results_df)}")
    print(f"Significant (raw p<0.05): {(results_df['p_value'] < 0.05).sum()}")
    print(f"Significant (FDR corrected): {results_df['significant_fdr'].sum()}")
    print(f"\nBy family:")
    for family in sorted(results_df["family"].unique()):
        fam_data = results_df[results_df["family"] == family]
        sig_fdr = fam_data["significant_fdr"].sum()
        print(f"  Family {family}: {len(fam_data)} tests, {sig_fdr} significant (FDR)")
    
    print("\n" + "="*80)
    print(f"Completed: {datetime.now().isoformat()}")
    print("="*80)

if __name__ == "__main__":
    main()
