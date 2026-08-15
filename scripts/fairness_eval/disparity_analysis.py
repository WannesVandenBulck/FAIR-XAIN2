#!/usr/bin/env python
"""
Disparity Analysis: Statistical tests comparing instance-level metrics across demographic groups.

Loads extraction JSON files and computes instance-level faithfulness metrics on-the-fly,
then stratifies by demographic group for statistical testing.

Uses Mann-Whitney U (binary comparisons).
Reports both raw p-values and multiple comparison corrections (Bonferroni and FDR).
Excludes groups with n < 5 samples.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import glob
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
DATASETS = ["credit", "law", "saudi", "student"]
DATASET_SIZES = {"credit": 97, "law": 308, "saudi": 106, "student": 73}
SIGNIFICANCE_THRESHOLD = 0.05

PROTECTED_ATTRS = {
    "credit":  ["age", "sex", "foreign_worker"],
    "law":     ["gender", "race"],
    "saudi":   ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
}

NUMERIC_ATTRS = {
    "credit":  ["age"],
    "law":     [],
    "saudi":   ["Age"],
    "student": ["age"],
}

EXTRACTIONS_DIR = ROOT / "results" / "extractions"
GT_DIR = ROOT / "results" / "ground_truth" / "json"
ADVERSE_DATA_PATH = ROOT / "datasets_prep" / "data"

OUTPUT_FILE = ROOT / "results" / "fairness_eval" / "disparity_analysis.csv"
SUMMARY_FILE = ROOT / "results" / "fairness_eval" / "disparity_significant_findings.txt"

# ============================================================

def load_demographic_data(dataset):
    """Load demographic attributes from adverse CSV."""
    adverse_file = ADVERSE_DATA_PATH / f"{dataset}_dataset" / f"{dataset}_adverse.csv"
    if not adverse_file.exists():
        print(f"Warning: {adverse_file} not found")
        return pd.DataFrame()
    
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


def load_extraction(dataset, instance_idx, narrative_provider, extractor_provider, condition):
    """Load extraction JSON for a single instance."""
    pattern = f"{EXTRACTIONS_DIR}/{dataset}/{condition}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.json"
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    try:
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def load_ground_truth(dataset, instance_idx):
    """Load ground truth JSON for a single instance."""
    gt_file = GT_DIR / dataset / f"instance_{instance_idx}.json"
    
    if not gt_file.exists():
        return None
    
    try:
        with open(gt_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def compute_faithfulness_metrics(extraction_data, gt_data):
    """
    Compute instance-level faithfulness metrics comparing extraction to ground truth.
    
    Returns dict with metrics comparing extracted features to ground truth SHAP features.
    """
    if not extraction_data or not gt_data:
        return None
    
    metrics = {}
    
    # Extract features from extraction JSON (has most_important_features array)
    extracted_features = {}
    if 'most_important_features' in extraction_data and extraction_data['most_important_features']:
        for feat in extraction_data['most_important_features']:
            rank = feat.get('rank', 0)
            name = str(feat.get('name', '')).lower().strip()
            if name and rank > 0:
                extracted_features[rank] = {
                    "name": name,
                    "sign": feat.get('sign'),
                }
    
    # Extract ground truth features (has SHAP_feature_N_name, SHAP_feature_N_sign)
    gt_features = {}
    for i in range(1, 4):
        key_name = f"SHAP_feature_{i}_name"
        if key_name in gt_data and gt_data[key_name]:
            feature_name = str(gt_data[key_name]).lower().strip()
            if feature_name:
                gt_features[i] = {
                    "name": feature_name,
                    "sign": gt_data.get(f"SHAP_feature_{i}_sign"),
                }
    
    # Rank accuracy: does extracted feature at rank N match GT feature at rank N?
    rank_accuracies = []
    for rank in [1, 2, 3]:
        if rank in extracted_features and rank in gt_features:
            match = (extracted_features[rank]["name"] == gt_features[rank]["name"])
            rank_accuracies.append(1.0 if match else 0.0)
        elif rank in gt_features:
            rank_accuracies.append(0.0)
    
    if rank_accuracies:
        metrics["rank1_accuracy"] = rank_accuracies[0] if len(rank_accuracies) > 0 else np.nan
        metrics["rank2_accuracy"] = rank_accuracies[1] if len(rank_accuracies) > 1 else np.nan
        metrics["rank3_accuracy"] = rank_accuracies[2] if len(rank_accuracies) > 2 else np.nan
        metrics["rank_total_accuracy"] = np.mean(rank_accuracies)
    
    # Sign accuracy: % of mentioned features with correct sign
    correct_signs = 0
    total_signed = 0
    for rank, feat in extracted_features.items():
        if rank in gt_features and feat.get("sign") is not None:
            total_signed += 1
            if feat["sign"] == gt_features[rank].get("sign"):
                correct_signs += 1
    
    metrics["sign_accuracy"] = correct_signs / total_signed if total_signed > 0 else np.nan
    
    # Probability accuracy
    if 'predicted_probability' in extraction_data and 'predicted_probability' in gt_data:
        extracted_prob = float(extraction_data['predicted_probability'])
        gt_prob = float(gt_data['predicted_probability'])
        prob_diff = abs(extracted_prob - gt_prob)
        metrics["probability_accuracy"] = 1.0 - min(prob_diff, 1.0)  # Scaled to [0, 1]
    
    return metrics


def apply_multiple_correction(p_values):
    """Apply Bonferroni and FDR corrections."""
    if not p_values:
        return {}
    
    p_array = np.array(p_values)
    
    # Bonferroni
    bonferroni = np.minimum(p_array * len(p_values), 1.0)
    
    # FDR (Benjamini-Hochberg)
    sorted_idx = np.argsort(p_array)
    sorted_p = p_array[sorted_idx]
    m = len(sorted_p)
    fdr_values = sorted_p * m / (np.arange(m) + 1)
    fdr_values = np.minimum.accumulate(fdr_values[::-1])[::-1]
    
    fdr = np.zeros_like(p_array)
    fdr[sorted_idx] = fdr_values
    
    return {
        'bonferroni': bonferroni,
        'fdr': fdr
    }


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def mann_whitney_u_test(group1, group2, group1_name, group2_name):
    """Perform Mann-Whitney U test."""
    if len(group1) < 5 or len(group2) < 5:
        return None
    
    try:
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        effect_size = cohens_d(group1, group2)
        mean_diff = np.mean(group1) - np.mean(group2)
        
        return {
            'group1': group1_name,
            'group2': group2_name,
            'n_group1': len(group1),
            'n_group2': len(group2),
            'mean_group1': np.mean(group1),
            'mean_group2': np.mean(group2),
            'mean_diff': mean_diff,
            'p_value': p_value,
            'cohens_d': effect_size,
            'statistic': statistic
        }
    except Exception:
        return None


def disparity_1_total(dataset, protected_attrs, numeric_attrs, demographics, demo_df, 
                      narrative_provider, extractor_provider):
    """Type 1: Total Disparity - Within include_pa, compare demographic groups."""
    results = []
    condition = "include_pa"
    
    # Collect instance-level metrics stratified by demographic group
    instance_metrics_by_attr = defaultdict(lambda: defaultdict(list))
    
    num_instances = DATASET_SIZES[dataset]
    for instance_idx in range(num_instances):
        extraction = load_extraction(dataset, instance_idx, narrative_provider, extractor_provider, condition)
        gt = load_ground_truth(dataset, instance_idx)
        
        if not extraction or not gt:
            continue
        
        metrics = compute_faithfulness_metrics(extraction, gt)
        if not metrics:
            continue
        
        # Determine demographic group for this instance
        if instance_idx not in demographics:
            continue
        
        # Store metrics per attribute and group
        for attr in protected_attrs:
            if attr not in demographics[instance_idx]:
                continue
            
            group_value = demographics[instance_idx][attr]
            
            for metric_name, metric_val in metrics.items():
                if not np.isnan(metric_val):
                    instance_metrics_by_attr[attr][group_value].append(metric_val)
    
    # Run Mann-Whitney U tests
    for attr in protected_attrs:
        if attr not in instance_metrics_by_attr:
            continue
        
        groups_dict = instance_metrics_by_attr[attr]
        group_values = sorted(list(groups_dict.keys()))
        
        if len(group_values) != 2:
            continue
        
        group1_val, group2_val = group_values[0], group_values[1]
        group1_data = np.array(groups_dict[group1_val])
        group2_data = np.array(groups_dict[group2_val])
        
        # Test on combined metrics
        test_result = mann_whitney_u_test(group1_data, group2_data, group1_val, group2_val)
        if test_result:
            test_result.update({
                'disparity_type': 'Total',
                'dataset': dataset,
                'condition': condition,
                'narrative_provider': narrative_provider,
                'extractor_provider': extractor_provider,
                'protected_attribute': attr,
                'metric': 'rank_and_sign_accuracy'
            })
            results.append(test_result)
    
    return results


def main():
    """Run disparity analysis on instance-level metrics from extraction JSONs."""
    
    print("\n" + "=" * 100)
    print("DISPARITY ANALYSIS (Instance-Level Metrics from Extractions)")
    print("=" * 100)
    
    all_results = []
    all_p_values = []
    
    for dataset in DATASETS:
        print(f"\nProcessing {dataset.upper()}...")
        
        # Load demographic data
        demo_df = load_demographic_data(dataset)
        if demo_df.empty:
            print(f"  Warning: No demographic data for {dataset}")
            continue
        
        demographics = get_demographic_groups(dataset, demo_df, PROTECTED_ATTRS[dataset], 
                                             NUMERIC_ATTRS[dataset])
        
        # Process all provider combinations
        providers = ["deepseek", "grok", "openai"]
        extractors = ["grok"]
        
        for narrative_provider in providers:
            for extractor_provider in extractors:
                print(f"  {narrative_provider} (narrative) -> {extractor_provider} (extractor)...", end=" ", flush=True)
                
                try:
                    results = disparity_1_total(dataset, PROTECTED_ATTRS[dataset], 
                                               NUMERIC_ATTRS[dataset], demographics, demo_df,
                                               narrative_provider, extractor_provider)
                    
                    if results:
                        all_results.extend(results)
                        all_p_values.extend([r['p_value'] for r in results])
                        print(f"Found {len(results)} comparisons")
                    else:
                        print("No comparisons")
                except Exception as e:
                    print(f"Error: {e}")
    
    if not all_results:
        print("\nNo results found. Check that extraction JSON files exist in:")
        print(f"  {EXTRACTIONS_DIR}")
        return
    
    # Apply multiple comparison corrections
    print("\nApplying multiple comparison corrections...")
    corrections = apply_multiple_correction(all_p_values)
    
    for i, result in enumerate(all_results):
        result['p_value_bonferroni'] = corrections['bonferroni'][i]
        result['p_value_fdr'] = corrections['fdr'][i]
        result['significant_raw'] = result['p_value'] < SIGNIFICANCE_THRESHOLD
        result['significant_bonferroni'] = result['p_value_bonferroni'] < SIGNIFICANCE_THRESHOLD
        result['significant_fdr'] = result['p_value_fdr'] < SIGNIFICANCE_THRESHOLD
    
    # Convert to DataFrame and save
    df_results = pd.DataFrame(all_results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved results to {OUTPUT_FILE.relative_to(ROOT)}")
    
    # Generate summary of significant findings
    sig_findings = []
    sig_findings.append("=" * 100)
    sig_findings.append("STATISTICALLY SIGNIFICANT FINDINGS (p < 0.05)")
    sig_findings.append("=" * 100)
    sig_findings.append("")
    
    # Raw p-values
    sig_raw = df_results[df_results['significant_raw']]
    sig_findings.append(f"RAW P-VALUES (without correction): {len(sig_raw)} significant findings")
    sig_findings.append("-" * 100)
    if not sig_raw.empty:
        for _, row in sig_raw.iterrows():
            sig_findings.append(
                f"{row['disparity_type']:15} | {row['dataset']:10} | {row['protected_attribute']:15} | "
                f"p={row['p_value']:.4f} | n1={row['n_group1']}, n2={row['n_group2']} | d={row['cohens_d']:.3f}"
            )
    else:
        sig_findings.append("No significant findings")
    sig_findings.append("")
    
    # Bonferroni-corrected
    sig_bonf = df_results[df_results['significant_bonferroni']]
    sig_findings.append(f"BONFERRONI CORRECTED: {len(sig_bonf)} significant findings")
    sig_findings.append("-" * 100)
    if not sig_bonf.empty:
        for _, row in sig_bonf.iterrows():
            sig_findings.append(
                f"{row['disparity_type']:15} | {row['dataset']:10} | {row['protected_attribute']:15} | "
                f"p={row['p_value_bonferroni']:.4f} | n1={row['n_group1']}, n2={row['n_group2']} | d={row['cohens_d']:.3f}"
            )
    else:
        sig_findings.append("No significant findings")
    sig_findings.append("")
    
    # FDR-corrected
    sig_fdr = df_results[df_results['significant_fdr']]
    sig_findings.append(f"FDR CORRECTED: {len(sig_fdr)} significant findings")
    sig_findings.append("-" * 100)
    if not sig_fdr.empty:
        for _, row in sig_fdr.iterrows():
            sig_findings.append(
                f"{row['disparity_type']:15} | {row['dataset']:10} | {row['protected_attribute']:15} | "
                f"p={row['p_value_fdr']:.4f} | n1={row['n_group1']}, n2={row['n_group2']} | d={row['cohens_d']:.3f}"
            )
    else:
        sig_findings.append("No significant findings")
    sig_findings.append("")
    sig_findings.append("=" * 100)
    
    # Save summary
    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(sig_findings))
    
    print(f"Saved summary to {SUMMARY_FILE.relative_to(ROOT)}")
    
    print("\n" + "\n".join(sig_findings))
    print(f"\nTotal comparisons: {len(all_results)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
