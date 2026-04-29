"""
Calculate SHAP metrics with flexible breakdowns:
1. All metrics by provider (all instances)
2. Sex breakdown averaged across providers
3. Sex breakdown for each provider individually

Usage:
    python scripts/shap_metrics_by_provider_and_protected_attrs.py
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


PROVIDERS = ["openai", "gemini", "grok", "deepseek", "mistral", "claude"]
DATASET_NAME = "credit"
NUM_INSTANCES = 34
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_ground_truth(instance_idx):
    """Load ground truth JSON file."""
    path = f"results/ground_truth/{DATASET_NAME}/instance_{instance_idx}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_extraction(provider, instance_idx):
    """Load majority voted extraction for a narrative provider.
    
    Loads the consensus extraction created by majority voting across
    the three extractor LLMs (deepseek, grok, openai).
    """
    path = f"results/extractions/majority/{provider}/instance_{instance_idx}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def calculate_rank_agreement(extracted_ranks, gt_ranks):
    """Calculate rank agreement for SHAP features.
    
    Always returns a value (0 or 1), never None.
    If either extraction or ground truth is missing, counts as disagreement (0).
    """
    agreements = []
    for i in range(3):
        if i < len(extracted_ranks) and i < len(gt_ranks):
            extracted_name = extracted_ranks[i]
            gt_name = gt_ranks[i]
            # Both present: check if they match
            agreement = 1.0 if extracted_name == gt_name else 0.0
            agreements.append(agreement)
        else:
            # Missing data = disagreement
            agreements.append(0.0)
    return agreements


def calculate_sign_agreement(extracted_signs, gt_signs):
    """Calculate sign agreement by matching feature names, not ranks.
    
    Compares signs for the same features regardless of their rank position.
    Returns a list of agreements (0 or 1) for each matched feature.
    """
    # This function is kept for backward compatibility but uses name-based comparison
    agreements = []
    return agreements


def calculate_total_sign_agreement_by_name(extraction, ground_truth):
    """Calculate total sign agreement by matching feature names, not ranks.
    
    Compares signs for the same features regardless of their rank position.
    Returns a list of agreements (0 or 1) for each matched feature.
    """
    agreements = []
    
    # Create lookup dicts by feature name
    extracted_features = {f.get("name"): f for f in extraction.get("most_important_features", [])}
    gt_features = {f.get("name"): f for f in ground_truth.get("most_important_features", [])}
    
    # Compare signs for features that appear in both
    for feature_name in extracted_features:
        if feature_name in gt_features:
            extracted_sign = extracted_features[feature_name].get("sign")
            gt_sign = gt_features[feature_name].get("sign")
            
            if extracted_sign is not None and gt_sign is not None:
                try:
                    if int(extracted_sign) == int(gt_sign):
                        agreements.append(1.0)
                    else:
                        agreements.append(0.0)
                except (ValueError, TypeError):
                    agreements.append(0.0)
            else:
                agreements.append(0.0)
    
    return agreements


def calculate_shap_value_agreement_by_name(extraction, ground_truth):
    """Calculate SHAP value agreement by matching feature names, not ranks.
    
    Compares values for the same features regardless of their rank position.
    Only compares when both extraction and ground truth have the value (not NaN).
    Returns a list of agreements (0 or 1) for each matched feature.
    """
    agreements = []
    
    # Create lookup dicts by feature name
    extracted_features = {f.get("name"): f for f in extraction.get("most_important_features", [])}
    gt_features = {f.get("name"): f for f in ground_truth.get("most_important_features", [])}
    
    # Compare values for features that appear in both
    for feature_name in extracted_features:
        if feature_name in gt_features:
            extracted_val = extracted_features[feature_name].get("value")
            gt_val = gt_features[feature_name].get("value")
            
            # Only compare if both values are present (not NaN)
            if extracted_val != "NaN" and gt_val != "NaN" and extracted_val is not None and gt_val is not None:
                if extracted_val == gt_val:
                    agreements.append(1.0)
                else:
                    agreements.append(0.0)
            # If either is NaN, skip this comparison entirely
    
    return agreements


def calculate_protected_attrs_value_agreement(extraction, ground_truth):
    """Calculate value agreement for protected attributes only."""
    agreements = []
    gt_features = {f.get("name"): f for f in ground_truth.get("features", [])}
    protected_attrs = ["sex", "age", "foreign_worker"]
    
    for feature in extraction.get("features", []):
        name = feature.get("name")
        if name in protected_attrs and name in gt_features:
            gt_feature = gt_features[name]
            if feature.get("value") != "NaN" and gt_feature.get("value") != "NaN":
                if feature.get("value") == gt_feature.get("value"):
                    agreements.append(1.0)
                else:
                    agreements.append(0.0)
    
    return agreements


def calculate_other_features_value_agreement(extraction, ground_truth):
    """Calculate value agreement for non-SHAP, non-protected feature values."""
    agreements = []
    gt_features = {f.get("name"): f for f in ground_truth.get("features", [])}
    protected_attrs = ["sex", "age", "foreign_worker"]
    shap_names = {f.get("name") for f in ground_truth.get("most_important_features", [])}
    
    for feature in extraction.get("features", []):
        name = feature.get("name")
        # Only count non-protected, non-SHAP features
        if name not in protected_attrs and name not in shap_names and name in gt_features:
            gt_feature = gt_features[name]
            if feature.get("value") != "NaN" and gt_feature.get("value") != "NaN":
                if feature.get("value") == gt_feature.get("value"):
                    agreements.append(1.0)
                else:
                    agreements.append(0.0)
    
    return agreements


def calculate_all_value_agreement(extraction, ground_truth):
    """Calculate agreement for all feature values (SHAP, protected, and other)."""
    agreements = []
    
    # Get SHAP value agreements (using name-based comparison)
    agreements.extend(calculate_shap_value_agreement_by_name(extraction, ground_truth))
    
    # Get protected attributes value agreements
    agreements.extend(calculate_protected_attrs_value_agreement(extraction, ground_truth))
    
    # Get other features value agreements
    agreements.extend(calculate_other_features_value_agreement(extraction, ground_truth))
    
    return agreements


def calculate_metrics(extraction, ground_truth):
    """Calculate all metrics comparing extraction to ground truth."""
    metrics = {"valid": False}
    
    if not extraction or not ground_truth:
        return metrics
    
    metrics["valid"] = True
    
    # Count features mentioned
    metrics["features_mentioned"] = sum(
        1 for f in extraction.get("features", [])
        if f.get("mentioned", 0) == 1 and f.get("name") not in ["sex", "age", "foreign_worker"]
    )
    
    # Count protected attributes mentioned
    metrics["protected_attrs_mentioned"] = sum(
        1 for f in extraction.get("features", [])
        if f.get("mentioned", 0) == 1 and f.get("name") in ["sex", "age", "foreign_worker"]
    )
    
    # Count feature values given - ONLY for features that were mentioned (and are not protected attrs)
    metrics["feature_values_given"] = sum(
        1 for f in extraction.get("features", [])
        if f.get("mentioned", 0) == 1 and f.get("value") != "NaN" and f.get("name") not in ["sex", "age", "foreign_worker"]
    )
    
    # Count SHAP features mentioned (how many top features did the LLM identify)
    metrics["shap_features_mentioned"] = len(extraction.get("most_important_features", []))
    
    # Count SHAP values given
    metrics["shap_values_given"] = sum(
        1 for f in extraction.get("most_important_features", [])
        if f.get("value") != "NaN"
    )
    
    # Count protected attributes values given - ONLY for protected attrs that were mentioned
    metrics["protected_attrs_values_given"] = sum(
        1 for f in extraction.get("features", [])
        if f.get("mentioned", 0) == 1 and f.get("value") != "NaN" and f.get("name") in ["sex", "age", "foreign_worker"]
    )
    

    
    # Extract SHAP feature NAMES (not ranks!) from extraction
    extracted_names = [f.get("name") for f in extraction.get("most_important_features", [])[:3]]
    extracted_signs = [f.get("sign") for f in extraction.get("most_important_features", [])[:3]]
    
    # Extract SHAP feature NAMES from ground truth
    gt_names = [f.get("name") for f in ground_truth.get("most_important_features", [])[:3]]
    gt_signs = [f.get("sign") for f in ground_truth.get("most_important_features", [])[:3]]
    
    # Calculate agreements (comparing names, not ranks)
    metrics["rank_agreements"] = calculate_rank_agreement(extracted_names, gt_names)
    metrics["total_sign_agreement"] = calculate_total_sign_agreement_by_name(extraction, ground_truth)
    metrics["shap_value_agreement"] = calculate_shap_value_agreement_by_name(extraction, ground_truth)
    metrics["protected_attrs_value_agreement"] = calculate_protected_attrs_value_agreement(extraction, ground_truth)
    metrics["other_features_value_agreement"] = calculate_other_features_value_agreement(extraction, ground_truth)
    metrics["all_value_agreement"] = calculate_all_value_agreement(extraction, ground_truth)
    
    return metrics


def compute_metrics_by_provider_and_sex():
    """
    Compute metrics broken down by:
    - Narrative provider
    - Sex (0=male, 1=female)
    
    Returns three dictionaries:
    1. results_by_provider: {provider: metrics}
    2. results_by_sex: {sex: metrics}
    3. results_by_provider_and_sex: {(provider, sex): metrics}
    """
    
    results_by_provider = {}
    results_by_sex = {}
    results_by_provider_and_sex = {}
    
    print("\n" + "=" * 100)
    print("COMPUTING METRICS BY PROVIDER AND SEX")
    print("=" * 100)
    
    # Iterate over all instances
    for instance_idx in range(NUM_INSTANCES):
        print(f"\nProcessing instance {instance_idx}...", end=" ")
        
        # Load ground truth
        gt = load_ground_truth(instance_idx)
        if not gt:
            print(f"[WARN] Ground truth not found")
            continue
        
        # Get sex (only care about sex, not foreign_worker)
        sex = None
        for feature in gt.get("features", []):
            if feature.get("name") == "sex":
                sex = feature.get("value")
                break
        
        if sex is None:
            print(f"[WARN] Sex not found")
            continue
        
        print(f"sex={sex}")
        
        # Process each narrative provider
        for provider in PROVIDERS:
            # Load the majority voted extraction for this provider+instance
            extraction = load_extraction(provider, instance_idx)
            if not extraction:
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(extraction, gt)
            if not metrics["valid"]:
                continue
            
            # Initialize keys if needed
            if provider not in results_by_provider:
                results_by_provider[provider] = create_empty_metrics_dict()
            
            if sex not in results_by_sex:
                results_by_sex[sex] = create_empty_metrics_dict()
            
            key = (provider, sex)
            if key not in results_by_provider_and_sex:
                results_by_provider_and_sex[key] = create_empty_metrics_dict()
            
            # Accumulate metrics for all three groupings
            accumulate_metrics(results_by_provider[provider], metrics)
            accumulate_metrics(results_by_sex[sex], metrics)
            accumulate_metrics(results_by_provider_and_sex[key], metrics)
    
    return results_by_provider, results_by_sex, results_by_provider_and_sex


def create_empty_metrics_dict():
    """Create empty metrics accumulator."""
    return {
        "instances_count": 0,
        "features_mentioned_sum": 0,
        "feature_values_given_sum": 0,
        "protected_attrs_mentioned_sum": 0,
        "shap_features_mentioned_sum": 0,
        "shap_values_given_sum": 0,
        "protected_attrs_values_given_sum": 0,
        "rank_1_sum": 0,
        "rank_1_count": 0,
        "rank_2_sum": 0,
        "rank_2_count": 0,
        "rank_3_sum": 0,
        "rank_3_count": 0,
        "rank_total_sum": 0,
        "rank_total_count": 0,
        "total_sign_agreement_sum": 0,
        "total_sign_agreement_count": 0,
        "shap_value_agreement_sum": 0,
        "shap_value_agreement_count": 0,
        "protected_attrs_value_agreement_sum": 0,
        "protected_attrs_value_agreement_count": 0,
        "other_features_value_agreement_sum": 0,
        "other_features_value_agreement_count": 0,
        "all_value_agreement_sum": 0,
        "all_value_agreement_count": 0,
    }


def accumulate_metrics(accumulator, metrics):
    """Add metrics to accumulator."""
    accumulator["instances_count"] += 1
    accumulator["features_mentioned_sum"] += metrics["features_mentioned"]
    accumulator["feature_values_given_sum"] += metrics["feature_values_given"]
    accumulator["protected_attrs_mentioned_sum"] += metrics["protected_attrs_mentioned"]
    accumulator["shap_features_mentioned_sum"] += metrics["shap_features_mentioned"]
    accumulator["shap_values_given_sum"] += metrics["shap_values_given"]
    accumulator["protected_attrs_values_given_sum"] += metrics["protected_attrs_values_given"]
    
    # Rank agreements - always has values 0 or 1, never None
    for i, agreement in enumerate(metrics["rank_agreements"]):
        accumulator[f"rank_{i+1}_sum"] += agreement
        accumulator[f"rank_{i+1}_count"] += 1
        accumulator["rank_total_sum"] += agreement
        accumulator["rank_total_count"] += 1
    
    # Total sign agreement by feature name
    for agreement in metrics["total_sign_agreement"]:
        accumulator["total_sign_agreement_sum"] += agreement
        accumulator["total_sign_agreement_count"] += 1
    
    # SHAP value agreements
    for agreement in metrics["shap_value_agreement"]:
        accumulator["shap_value_agreement_sum"] += agreement
        accumulator["shap_value_agreement_count"] += 1
    
    # Protected attributes value agreements
    for agreement in metrics["protected_attrs_value_agreement"]:
        accumulator["protected_attrs_value_agreement_sum"] += agreement
        accumulator["protected_attrs_value_agreement_count"] += 1
    
    # Other features value agreements
    for agreement in metrics["other_features_value_agreement"]:
        accumulator["other_features_value_agreement_sum"] += agreement
        accumulator["other_features_value_agreement_count"] += 1
    
    # All value agreements
    for agreement in metrics["all_value_agreement"]:
        accumulator["all_value_agreement_sum"] += agreement
        accumulator["all_value_agreement_count"] += 1


def save_to_excel_by_provider(results_by_provider):
    """Save results by provider to Excel."""
    # Ensure output directory exists
    os.makedirs("results/shap_metrics", exist_ok=True)
    
    rows = []
    
    for provider in sorted(results_by_provider.keys()):
        metrics = results_by_provider[provider]
        
        if metrics["instances_count"] == 0:
            continue
        
        count = metrics["instances_count"]
        
        row = {
            "provider": provider,
            "instances_count": count,
            "avg_features_mentioned": metrics["features_mentioned_sum"] / count,
            "avg_feature_values_given": metrics["feature_values_given_sum"] / count,
            "avg_protected_attrs_mentioned": metrics["protected_attrs_mentioned_sum"] / count,
            "avg_shap_features_mentioned": metrics["shap_features_mentioned_sum"] / count,
            "avg_shap_values_given": metrics["shap_values_given_sum"] / count,
            "avg_protected_attrs_values_given": metrics["protected_attrs_values_given_sum"] / count,
            "rank_1_agreement_%": (metrics["rank_1_sum"] / metrics["rank_1_count"] * 100) if metrics["rank_1_count"] > 0 else None,
            "rank_2_agreement_%": (metrics["rank_2_sum"] / metrics["rank_2_count"] * 100) if metrics["rank_2_count"] > 0 else None,
            "rank_3_agreement_%": (metrics["rank_3_sum"] / metrics["rank_3_count"] * 100) if metrics["rank_3_count"] > 0 else None,
            "rank_total_agreement_%": (metrics["rank_total_sum"] / metrics["rank_total_count"] * 100) if metrics["rank_total_count"] > 0 else None,
            "total_sign_agreement_%": (metrics["total_sign_agreement_sum"] / metrics["total_sign_agreement_count"] * 100) if metrics["total_sign_agreement_count"] > 0 else None,
            "shap_value_agreement_%": (metrics["shap_value_agreement_sum"] / metrics["shap_value_agreement_count"] * 100) if metrics["shap_value_agreement_count"] > 0 else None,
            "protected_attrs_value_agreement_%": (metrics["protected_attrs_value_agreement_sum"] / metrics["protected_attrs_value_agreement_count"] * 100) if metrics["protected_attrs_value_agreement_count"] > 0 else None,
            "other_features_value_agreement_%": (metrics["other_features_value_agreement_sum"] / metrics["other_features_value_agreement_count"] * 100) if metrics["other_features_value_agreement_count"] > 0 else None,
            "all_value_agreement_%": (metrics["all_value_agreement_sum"] / metrics["all_value_agreement_count"] * 100) if metrics["all_value_agreement_count"] > 0 else None,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    output_path = f"results/shap_metrics/shap_metrics_by_provider_{DATASET_NAME}_{TIMESTAMP}.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"\n[OK] Saved: {output_path}")
    print(df.to_string())


def save_to_excel_by_sex_averaged(results_by_sex):
    """Save results by sex, averaged across providers."""
    rows = []
    sex_names = {0: "male", 1: "female"}
    
    for sex in sorted(results_by_sex.keys()):
        metrics = results_by_sex[sex]
        
        if metrics["instances_count"] == 0:
            continue
        
        count = metrics["instances_count"]
        
        row = {
            "sex": sex_names.get(sex, sex),
            "instances_count": count,
            "avg_features_mentioned": metrics["features_mentioned_sum"] / count,
            "avg_feature_values_given": metrics["feature_values_given_sum"] / count,
            "avg_protected_attrs_mentioned": metrics["protected_attrs_mentioned_sum"] / count,
            "avg_shap_features_mentioned": metrics["shap_features_mentioned_sum"] / count,
            "avg_shap_values_given": metrics["shap_values_given_sum"] / count,
            "avg_protected_attrs_values_given": metrics["protected_attrs_values_given_sum"] / count,
            "rank_1_agreement_%": (metrics["rank_1_sum"] / metrics["rank_1_count"] * 100) if metrics["rank_1_count"] > 0 else None,
            "rank_2_agreement_%": (metrics["rank_2_sum"] / metrics["rank_2_count"] * 100) if metrics["rank_2_count"] > 0 else None,
            "rank_3_agreement_%": (metrics["rank_3_sum"] / metrics["rank_3_count"] * 100) if metrics["rank_3_count"] > 0 else None,
            "rank_total_agreement_%": (metrics["rank_total_sum"] / metrics["rank_total_count"] * 100) if metrics["rank_total_count"] > 0 else None,
            "total_sign_agreement_%": (metrics["total_sign_agreement_sum"] / metrics["total_sign_agreement_count"] * 100) if metrics["total_sign_agreement_count"] > 0 else None,
            "shap_value_agreement_%": (metrics["shap_value_agreement_sum"] / metrics["shap_value_agreement_count"] * 100) if metrics["shap_value_agreement_count"] > 0 else None,
            "protected_attrs_value_agreement_%": (metrics["protected_attrs_value_agreement_sum"] / metrics["protected_attrs_value_agreement_count"] * 100) if metrics["protected_attrs_value_agreement_count"] > 0 else None,
            "other_features_value_agreement_%": (metrics["other_features_value_agreement_sum"] / metrics["other_features_value_agreement_count"] * 100) if metrics["other_features_value_agreement_count"] > 0 else None,
            "all_value_agreement_%": (metrics["all_value_agreement_sum"] / metrics["all_value_agreement_count"] * 100) if metrics["all_value_agreement_count"] > 0 else None,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    output_path = f"results/shap_metrics/shap_metrics_by_sex_averaged_{DATASET_NAME}_{TIMESTAMP}.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"\n[OK] Saved: {output_path}")
    print(df.to_string())


def save_to_excel_by_sex_and_provider(results_by_provider_and_sex):
    """Save results by sex for each provider individually."""
    rows = []
    sex_names = {0: "male", 1: "female"}
    
    for (provider, sex) in sorted(results_by_provider_and_sex.keys()):
        metrics = results_by_provider_and_sex[(provider, sex)]
        
        if metrics["instances_count"] == 0:
            continue
        
        count = metrics["instances_count"]
        
        row = {
            "provider": provider,
            "sex": sex_names.get(sex, sex),
            "instances_count": count,
            "avg_features_mentioned": metrics["features_mentioned_sum"] / count,
            "avg_feature_values_given": metrics["feature_values_given_sum"] / count,
            "avg_protected_attrs_mentioned": metrics["protected_attrs_mentioned_sum"] / count,
            "avg_shap_features_mentioned": metrics["shap_features_mentioned_sum"] / count,
            "avg_shap_values_given": metrics["shap_values_given_sum"] / count,
            "avg_protected_attrs_values_given": metrics["protected_attrs_values_given_sum"] / count,
            "rank_1_agreement_%": (metrics["rank_1_sum"] / metrics["rank_1_count"] * 100) if metrics["rank_1_count"] > 0 else None,
            "rank_2_agreement_%": (metrics["rank_2_sum"] / metrics["rank_2_count"] * 100) if metrics["rank_2_count"] > 0 else None,
            "rank_3_agreement_%": (metrics["rank_3_sum"] / metrics["rank_3_count"] * 100) if metrics["rank_3_count"] > 0 else None,
            "rank_total_agreement_%": (metrics["rank_total_sum"] / metrics["rank_total_count"] * 100) if metrics["rank_total_count"] > 0 else None,
            "total_sign_agreement_%": (metrics["total_sign_agreement_sum"] / metrics["total_sign_agreement_count"] * 100) if metrics["total_sign_agreement_count"] > 0 else None,
            "shap_value_agreement_%": (metrics["shap_value_agreement_sum"] / metrics["shap_value_agreement_count"] * 100) if metrics["shap_value_agreement_count"] > 0 else None,
            "protected_attrs_value_agreement_%": (metrics["protected_attrs_value_agreement_sum"] / metrics["protected_attrs_value_agreement_count"] * 100) if metrics["protected_attrs_value_agreement_count"] > 0 else None,
            "other_features_value_agreement_%": (metrics["other_features_value_agreement_sum"] / metrics["other_features_value_agreement_count"] * 100) if metrics["other_features_value_agreement_count"] > 0 else None,
            "all_value_agreement_%": (metrics["all_value_agreement_sum"] / metrics["all_value_agreement_count"] * 100) if metrics["all_value_agreement_count"] > 0 else None,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    output_path = f"results/shap_metrics/shap_metrics_by_sex_and_provider_{DATASET_NAME}_{TIMESTAMP}.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"\n[OK] Saved: {output_path}")
    print(df.to_string())


if __name__ == "__main__":
    results_by_provider, results_by_sex, results_by_provider_and_sex = compute_metrics_by_provider_and_sex()
    
    print(f"\n{'='*100}")
    print("GENERATING OUTPUT EXCEL FILES")
    print(f"{'='*100}")
    
    print("\n[1] ALL METRICS BY PROVIDER (all instances)")
    save_to_excel_by_provider(results_by_provider)
    
    print("\n[2] SEX BREAKDOWN AVERAGED ACROSS PROVIDERS")
    save_to_excel_by_sex_averaged(results_by_sex)
    
    print("\n[3] SEX BREAKDOWN FOR EACH PROVIDER INDIVIDUALLY")
    save_to_excel_by_sex_and_provider(results_by_provider_and_sex)
