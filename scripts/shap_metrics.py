"""
Calculate SHAP narrative extraction metrics and compare with ground truth.

Metrics computed:
(a) Average features mentioned (non-SHAP, non-protected)
(b) Average feature values given (total across all feature types)
(c) Average protected attributes mentioned
(d) SHAP rank agreement - separate column for each rank (1, 2, 3)
(e) SHAP sign agreement - separate column for each rank (1, 2, 3)
(f) SHAP value agreement - for top SHAP features
(g) Predicted probability mention % and agreement %
(h) Overall value agreement % - for all extracted values against ground truth
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import glob


DATASETS = {
    "credit": {
        "protected_attrs": ["age", "sex", "foreign_worker"],
        "num_shap_features": 3,
        "num_instances": 34
    },
    "law": {
        "protected_attrs": ["gender", "race"],
        "num_shap_features": 3,
        "num_instances": 308
    }
}


def load_extraction_json(dataset_name, instance_idx, narrative_provider, extractor_provider, prompt_type="shap"):
    """Load extraction JSON file."""
    path = f"results/extractions/{dataset_name}/extractions/{prompt_type}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_ground_truth_json(dataset_name, instance_idx):
    """Load ground truth JSON file."""
    path = f"results/ground_truth/{dataset_name}/instance_{instance_idx}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_shap_feature_names(ground_truth):
    """Extract the top SHAP feature names from ground truth."""
    shap_names = []
    for feature in ground_truth.get("most_important_features", []):
        shap_names.append(feature["name"])
    return shap_names


def calculate_metrics(extraction, ground_truth, dataset_name):
    """
    Calculate all metrics comparing extraction with ground truth.
    Returns a dict with all metric values including per-rank metrics.
    """
    metrics = {
        "valid": False,
        "features_mentioned": 0,           # (a) non-SHAP, non-protected features mentioned
        "feature_values_given": 0,         # (b) total feature values given
        "protected_attrs_mentioned": 0,    # (c) protected attributes mentioned
        "shap_values_given": 0,            # count of SHAP features with values given
        "shap_feature_values_given": 0,    # NEW: count of SHAP feature values
        "protected_attrs_values_given": 0, # NEW: count of protected attribute values
        "other_feature_values_given": 0,   # NEW: count of other feature values (not SHAP, not protected)
        "rank_agreements": [None, None, None],      # (d) per-rank agreement for ranks 1,2,3
        "sign_agreements": [None, None, None],      # (e) per-rank sign agreement for ranks 1,2,3
        "shap_value_agreement": [],        # (f) value agreement for top SHAP features
        "all_value_agreement": [],         # (h) overall value agreement for all extracted values
        "predicted_prob_mentioned": False, # (g) whether prob was mentioned
        "predicted_prob_agreement": False, # (g) whether prob matches ground truth
    }
    
    if not extraction or not ground_truth:
        return metrics
    
    metrics["valid"] = True
    
    # Get SHAP feature names from ground truth
    shap_feature_names = get_shap_feature_names(ground_truth)
    protected_attrs = DATASETS[dataset_name]["protected_attrs"]
    
    # ========== (a) Count non-SHAP, non-protected features mentioned ==========
    extraction_features = extraction.get("features", [])
    for feat in extraction_features:
        if feat["name"] not in shap_feature_names and feat["name"] not in protected_attrs:
            if feat.get("mentioned") == 1:
                metrics["features_mentioned"] += 1
    
    # ========== (b) Count all feature values given (across all feature types) ==========
    for feat in extraction_features:
        if feat.get("value") and feat.get("value") != "NaN":
            metrics["feature_values_given"] += 1
            
            # NEW: Track by category
            if feat.get("name") in shap_feature_names:
                metrics["shap_feature_values_given"] += 1
            elif feat.get("name") in protected_attrs:
                metrics["protected_attrs_values_given"] += 1
            else:
                metrics["other_feature_values_given"] += 1
    
    # ========== (c) Count protected attributes mentioned ==========
    for feat in extraction_features:
        if feat["name"] in protected_attrs and feat.get("mentioned") == 1:
            metrics["protected_attrs_mentioned"] += 1
    
    # ========== NEW: Count SHAP features with values given ==========
    for feat in extraction_features:
        if feat["name"] in shap_feature_names:
            if feat.get("value") and feat.get("value") != "NaN":
                metrics["shap_values_given"] += 1
    
    # ========== (d) & (e) SHAP metrics: rank and sign per position ==========
    extraction_shap = extraction.get("most_important_features", [])
    ground_truth_shap = ground_truth.get("most_important_features", [])
    
    # ========== (d) & (e) SHAP metrics: rank and sign per position ==========
    extraction_shap = extraction.get("most_important_features", [])
    ground_truth_shap = ground_truth.get("most_important_features", [])
    
    # Create maps for easy lookup by rank and by name
    gt_by_rank = {feat.get("rank"): feat for feat in ground_truth_shap}
    gt_by_name = {feat.get("name"): feat for feat in ground_truth_shap}
    
    for rank_pos in range(1, 4):  # Ranks 1, 2, 3
        ext_feat = next((f for f in extraction_shap if f.get("rank") == rank_pos), None)
        gt_feat_at_rank = gt_by_rank.get(rank_pos)
        
        # ===== RANK AGREEMENT: Compare feature names at same rank position =====
        rank_agreement = False
        if ext_feat and gt_feat_at_rank:
            if ext_feat.get("name") == gt_feat_at_rank.get("name"):
                metrics["rank_agreements"][rank_pos - 1] = 1
                rank_agreement = True
            else:
                metrics["rank_agreements"][rank_pos - 1] = 0
        else:
            metrics["rank_agreements"][rank_pos - 1] = 0
        
        # ===== SIGN AGREEMENT: Only check when rank agreement succeeded OR when we can match by name =====
        # If rank matches, compare the sign at that position
        # If rank doesn't match, still try to find the feature by name and compare sign
        if ext_feat:
            ext_feat_name = ext_feat.get("name")
            
            # First try: if rank agreed, use the feature at the same rank
            if rank_agreement:
                gt_feat_to_compare = gt_feat_at_rank
            else:
                # If rank didn't agree, try to find the feature by name anywhere in ground truth
                gt_feat_to_compare = gt_by_name.get(ext_feat_name)
            
            if gt_feat_to_compare:
                ext_sign = ext_feat.get("sign")
                gt_sign = gt_feat_to_compare.get("sign")
                if ext_sign is not None and gt_sign is not None:
                    try:
                        ext_sign_int = int(ext_sign)
                        gt_sign_int = int(gt_sign)
                        metrics["sign_agreements"][rank_pos - 1] = 1 if ext_sign_int == gt_sign_int else 0
                    except (ValueError, TypeError):
                        metrics["sign_agreements"][rank_pos - 1] = 0
                else:
                    metrics["sign_agreements"][rank_pos - 1] = 0
            else:
                metrics["sign_agreements"][rank_pos - 1] = 0
        else:
            metrics["sign_agreements"][rank_pos - 1] = 0
        
        # ===== VALUE AGREEMENT: Same logic as sign agreement =====
        if ext_feat:
            ext_feat_name = ext_feat.get("name")
            
            # First try: if rank agreed, use the feature at the same rank
            if rank_agreement:
                gt_feat_to_compare = gt_feat_at_rank
            else:
                # If rank didn't agree, try to find the feature by name anywhere in ground truth
                gt_feat_to_compare = gt_by_name.get(ext_feat_name)
            
            if gt_feat_to_compare:
                ext_value = ext_feat.get("value")
                gt_value = gt_feat_to_compare.get("value")
                if ext_value and ext_value != "NaN" and gt_value and gt_value != "NaN":
                    try:
                        ext_val_num = float(ext_value)
                        gt_val_num = float(gt_value)
                        metrics["shap_value_agreement"].append(abs(ext_val_num - gt_val_num) < 0.1)
                    except (ValueError, TypeError):
                        metrics["shap_value_agreement"].append(ext_value == gt_value)
                        metrics["shap_value_agreement"].append(ext_value == gt_value)
    
    # ========== Overall value agreement for all extracted values ==========
    ground_truth_features = {f["name"]: f for f in ground_truth.get("features", [])}
    for ext_feat in extraction_features:
        feat_name = ext_feat["name"]
        if feat_name in ground_truth_features:
            ext_value = ext_feat.get("value")
            gt_value = ground_truth_features[feat_name].get("value")
            
            if ext_value and ext_value != "NaN" and gt_value and gt_value != "NaN":
                try:
                    ext_val_num = float(ext_value)
                    gt_val_num = float(gt_value)
                    metrics["all_value_agreement"].append(abs(ext_val_num - gt_val_num) < 0.1)
                except (ValueError, TypeError):
                    metrics["all_value_agreement"].append(ext_value == gt_value)
    
    # ========== (g) Predicted probability ==========
    ext_prob = extraction.get("predicted_probability")
    gt_prob = ground_truth.get("predicted_probability")
    
    if ext_prob and ext_prob != "NaN":
        metrics["predicted_prob_mentioned"] = True
        if gt_prob and gt_prob != "NaN":
            try:
                ext_prob_num = float(ext_prob)
                gt_prob_num = float(gt_prob)
                metrics["predicted_prob_agreement"] = abs(ext_prob_num - gt_prob_num) < 0.05
            except (ValueError, TypeError):
                metrics["predicted_prob_agreement"] = ext_prob == gt_prob
    
    return metrics


def aggregate_metrics_by_provider(dataset_name, prompt_type="shap"):
    """
    Aggregate metrics by NARRATIVE PROVIDER (rows are narrative generators, not extractors).
    Returns dict with narrative_provider -> aggregated metrics.
    Dynamically finds all extraction files that exist.
    """
    # Find all MAJORITY VOTED extraction files
    extraction_pattern = f"results/extractions/{dataset_name}/extractions/{prompt_type}/*/majority_voted/instance_*.json"
    extraction_files = glob.glob(extraction_pattern)
    
    if not extraction_files:
        print(f"[WARN] No majority voted extraction files found for {dataset_name}")
        return {}
    
    print(f"Found {len(extraction_files)} majority voted extraction files for {dataset_name}")
    
    # Parse extraction files to find which providers/narratives/instances exist
    extraction_map = defaultdict(list)  # narrative_provider -> list of (extractor_provider, instance_idx)
    
    for file_path in extraction_files:
        # Extract info from path: results/extractions/{dataset}/extractions/{prompt}/{narrative}/majority_voted/instance_{idx}.json
        parts = file_path.split(os.sep)
        try:
            narrative_provider = parts[-3]
            extractor_provider = "majority_voted"  # Always majority_voted now
            instance_idx = int(parts[-1].replace("instance_", "").replace(".json", ""))
            extraction_map[narrative_provider].append((extractor_provider, instance_idx))
        except (IndexError, ValueError):
            continue
    
    print(f"Available narrative providers: {sorted(extraction_map.keys())}")
    
    results = {}
    
    for narrative_provider in sorted(extraction_map.keys()):
        provider_metrics = {
            "extractions_processed": 0,
            "valid_instances": 0,
            
            # (a), (b), (c) averages
            "features_mentioned_sum": 0,
            "feature_values_given_sum": 0,
            "protected_attrs_mentioned_sum": 0,
            "shap_values_given_sum": 0,  # NEW: sum of SHAP values given
            "shap_feature_values_given_sum": 0,  # NEW: sum of SHAP feature values
            "protected_attrs_values_given_sum": 0,  # NEW: sum of protected attribute values
            "other_feature_values_given_sum": 0,  # NEW: sum of other feature values
            
            # (d) rank agreement per rank
            "rank_1_agreement_sum": 0,
            "rank_1_agreement_count": 0,
            "rank_2_agreement_sum": 0,
            "rank_2_agreement_count": 0,
            "rank_3_agreement_sum": 0,
            "rank_3_agreement_count": 0,
            "rank_total_agreement_sum": 0,  # NEW: total across all ranks
            "rank_total_agreement_count": 0,  # NEW: total count across all ranks
            
            # (e) sign agreement per rank
            "sign_1_agreement_sum": 0,
            "sign_1_agreement_count": 0,
            "sign_2_agreement_sum": 0,
            "sign_2_agreement_count": 0,
            "sign_3_agreement_sum": 0,
            "sign_3_agreement_count": 0,
            "sign_total_agreement_sum": 0,  # NEW: total across all ranks
            "sign_total_agreement_count": 0,  # NEW: total count across all ranks
            
            # (f) SHAP value agreement
            "shap_value_agreement_sum": 0,
            "shap_value_agreement_count": 0,
            
            # (g) Predicted probability
            "predicted_prob_mentioned_count": 0,
            "predicted_prob_agreement_count": 0,
            
            # (h) Overall value agreement
            "all_value_agreement_sum": 0,
            "all_value_agreement_count": 0,
        }
        
        # Get all extractor/instance combinations for this narrative provider
        extractor_instance_pairs = extraction_map[narrative_provider]
        
        for extractor_provider, instance_idx in extractor_instance_pairs:
            extraction = load_extraction_json(dataset_name, instance_idx, narrative_provider, extractor_provider, prompt_type)
            ground_truth = load_ground_truth_json(dataset_name, instance_idx)
            
            if not extraction or not ground_truth:
                continue
            
            provider_metrics["extractions_processed"] += 1
            metrics = calculate_metrics(extraction, ground_truth, dataset_name)
            
            if metrics["valid"]:
                provider_metrics["valid_instances"] += 1
                
                # (a), (b), (c), NEW
                provider_metrics["features_mentioned_sum"] += metrics["features_mentioned"]
                provider_metrics["feature_values_given_sum"] += metrics["feature_values_given"]
                provider_metrics["protected_attrs_mentioned_sum"] += metrics["protected_attrs_mentioned"]
                provider_metrics["shap_values_given_sum"] += metrics["shap_values_given"]
                provider_metrics["shap_feature_values_given_sum"] += metrics["shap_feature_values_given"]
                provider_metrics["protected_attrs_values_given_sum"] += metrics["protected_attrs_values_given"]
                provider_metrics["other_feature_values_given_sum"] += metrics["other_feature_values_given"]
                
                # (d) rank agreements per position
                for rank_pos in range(3):
                    if metrics["rank_agreements"][rank_pos] is not None:
                        # NEW: track total across all ranks
                        provider_metrics["rank_total_agreement_sum"] += metrics["rank_agreements"][rank_pos]
                        provider_metrics["rank_total_agreement_count"] += 1
                        
                        if rank_pos == 0:
                            provider_metrics["rank_1_agreement_sum"] += metrics["rank_agreements"][rank_pos]
                            provider_metrics["rank_1_agreement_count"] += 1
                        elif rank_pos == 1:
                            provider_metrics["rank_2_agreement_sum"] += metrics["rank_agreements"][rank_pos]
                            provider_metrics["rank_2_agreement_count"] += 1
                        elif rank_pos == 2:
                            provider_metrics["rank_3_agreement_sum"] += metrics["rank_agreements"][rank_pos]
                            provider_metrics["rank_3_agreement_count"] += 1
                
                # (e) sign agreements per position
                for rank_pos in range(3):
                    if metrics["sign_agreements"][rank_pos] is not None:
                        # NEW: track total across all ranks
                        provider_metrics["sign_total_agreement_sum"] += metrics["sign_agreements"][rank_pos]
                        provider_metrics["sign_total_agreement_count"] += 1
                        
                        if rank_pos == 0:
                            provider_metrics["sign_1_agreement_sum"] += metrics["sign_agreements"][rank_pos]
                            provider_metrics["sign_1_agreement_count"] += 1
                        elif rank_pos == 1:
                            provider_metrics["sign_2_agreement_sum"] += metrics["sign_agreements"][rank_pos]
                            provider_metrics["sign_2_agreement_count"] += 1
                        elif rank_pos == 2:
                            provider_metrics["sign_3_agreement_sum"] += metrics["sign_agreements"][rank_pos]
                            provider_metrics["sign_3_agreement_count"] += 1
                
                # (f) SHAP value agreement
                for agreement in metrics["shap_value_agreement"]:
                    provider_metrics["shap_value_agreement_sum"] += agreement
                    provider_metrics["shap_value_agreement_count"] += 1
                
                # (g) Predicted probability
                if metrics["predicted_prob_mentioned"]:
                    provider_metrics["predicted_prob_mentioned_count"] += 1
                if metrics["predicted_prob_agreement"]:
                    provider_metrics["predicted_prob_agreement_count"] += 1
                
                # (h) Overall value agreement
                for agreement in metrics["all_value_agreement"]:
                    provider_metrics["all_value_agreement_sum"] += agreement
                    provider_metrics["all_value_agreement_count"] += 1
        
        # Calculate averages and percentages
        if provider_metrics["valid_instances"] > 0:
            results[narrative_provider] = {
                "provider": narrative_provider,
                "extractions_processed": provider_metrics["extractions_processed"],
                
                # (a), (b), (c), NEW
                "avg_features_mentioned": provider_metrics["features_mentioned_sum"] / provider_metrics["valid_instances"],
                "avg_feature_values_given": provider_metrics["feature_values_given_sum"] / provider_metrics["valid_instances"],
                "avg_protected_attrs_mentioned": provider_metrics["protected_attrs_mentioned_sum"] / provider_metrics["valid_instances"],
                "avg_shap_values_given": provider_metrics["shap_values_given_sum"] / provider_metrics["valid_instances"],
                "avg_shap_feature_values_given": provider_metrics["shap_feature_values_given_sum"] / provider_metrics["valid_instances"],
                "avg_protected_attrs_values_given": provider_metrics["protected_attrs_values_given_sum"] / provider_metrics["valid_instances"],
                "avg_other_feature_values_given": provider_metrics["other_feature_values_given_sum"] / provider_metrics["valid_instances"],
                
                # (d) rank agreement %
                "rank_1_agreement_pct": (provider_metrics["rank_1_agreement_sum"] / provider_metrics["rank_1_agreement_count"] * 100) if provider_metrics["rank_1_agreement_count"] > 0 else None,
                "rank_2_agreement_pct": (provider_metrics["rank_2_agreement_sum"] / provider_metrics["rank_2_agreement_count"] * 100) if provider_metrics["rank_2_agreement_count"] > 0 else None,
                "rank_3_agreement_pct": (provider_metrics["rank_3_agreement_sum"] / provider_metrics["rank_3_agreement_count"] * 100) if provider_metrics["rank_3_agreement_count"] > 0 else None,
                "rank_total_agreement_pct": (provider_metrics["rank_total_agreement_sum"] / provider_metrics["rank_total_agreement_count"] * 100) if provider_metrics["rank_total_agreement_count"] > 0 else None,
                
                # (e) sign agreement %
                "sign_1_agreement_pct": (provider_metrics["sign_1_agreement_sum"] / provider_metrics["sign_1_agreement_count"] * 100) if provider_metrics["sign_1_agreement_count"] > 0 else None,
                "sign_2_agreement_pct": (provider_metrics["sign_2_agreement_sum"] / provider_metrics["sign_2_agreement_count"] * 100) if provider_metrics["sign_2_agreement_count"] > 0 else None,
                "sign_3_agreement_pct": (provider_metrics["sign_3_agreement_sum"] / provider_metrics["sign_3_agreement_count"] * 100) if provider_metrics["sign_3_agreement_count"] > 0 else None,
                "sign_total_agreement_pct": (provider_metrics["sign_total_agreement_sum"] / provider_metrics["sign_total_agreement_count"] * 100) if provider_metrics["sign_total_agreement_count"] > 0 else None,
                
                # (f) SHAP value agreement %
                "shap_value_agreement_pct": (provider_metrics["shap_value_agreement_sum"] / provider_metrics["shap_value_agreement_count"] * 100) if provider_metrics["shap_value_agreement_count"] > 0 else None,
                
                # (g) Predicted probability
                "predicted_prob_mention_pct": (provider_metrics["predicted_prob_mentioned_count"] / provider_metrics["valid_instances"] * 100),
                "predicted_prob_agreement_pct": (provider_metrics["predicted_prob_agreement_count"] / provider_metrics["predicted_prob_mentioned_count"] * 100) if provider_metrics["predicted_prob_mentioned_count"] > 0 else None,
                
                # (h) Overall value agreement %
                "all_value_agreement_pct": (provider_metrics["all_value_agreement_sum"] / provider_metrics["all_value_agreement_count"] * 100) if provider_metrics["all_value_agreement_count"] > 0 else None,
            }
    
    return results


def print_summary(results_by_provider, dataset_name):
    """Print summary of metrics by NARRATIVE PROVIDER with detailed columns."""
    print("\n" + "=" * 280)
    print(f"SHAP NARRATIVE EXTRACTION METRICS - {dataset_name.upper()} DATASET (by Narrative Provider)")
    print("=" * 280)
    
    # Print header line 1: Main categories
    header1 = (
        f"{'Provider':<12} {'Extr.':<6} "
        f"{'(a) Features':<12} {'(b) Values Breakdown':<52} {'(c) Prot.Attrs':<14} {'Avg SHAP Val':<12} "
        f"{'(d) Rank %':<26} {'(e) Sign %':<24} "
        f"{'(f) SHAP Val %':<14} {'(g) Prob':<24} {'(h) Val Agr %':<12}"
    )
    print(header1)
    
    # Print header line 2: Sub-breakdown
    header2 = (
        f"{'':12} {'':6} "
        f"{'Avg':<12} {'SHAP':<12} {'Prot.Attrs':<12} {'Other':<12} {'':12} {'Avg':<14} "
        f"{'R1%':<6} {'R2%':<6} {'R3%':<6} {'Total':<6} {'S1%':<6} {'S2%':<6} {'S3%':<6} {'Total':<6} "
        f"{'%':<14} {'Ment%':<12} {'Agr%':<12} {'%':<12}"
    )
    print(header2)
    print("-" * 280)
    
    for provider in sorted(results_by_provider.keys()):
        m = results_by_provider[provider]
        
        # Format percentages
        rank_1 = f"{m['rank_1_agreement_pct']:.0f}" if m['rank_1_agreement_pct'] is not None else "N/A"
        rank_2 = f"{m['rank_2_agreement_pct']:.0f}" if m['rank_2_agreement_pct'] is not None else "N/A"
        rank_3 = f"{m['rank_3_agreement_pct']:.0f}" if m['rank_3_agreement_pct'] is not None else "N/A"
        rank_total = f"{m['rank_total_agreement_pct']:.0f}" if m['rank_total_agreement_pct'] is not None else "N/A"
        
        sign_1 = f"{m['sign_1_agreement_pct']:.0f}" if m['sign_1_agreement_pct'] is not None else "N/A"
        sign_2 = f"{m['sign_2_agreement_pct']:.0f}" if m['sign_2_agreement_pct'] is not None else "N/A"
        sign_3 = f"{m['sign_3_agreement_pct']:.0f}" if m['sign_3_agreement_pct'] is not None else "N/A"
        sign_total = f"{m['sign_total_agreement_pct']:.0f}" if m['sign_total_agreement_pct'] is not None else "N/A"
        
        shap_val = f"{m['shap_value_agreement_pct']:.0f}" if m['shap_value_agreement_pct'] is not None else "N/A"
        prob_ment = f"{m['predicted_prob_mention_pct']:.0f}"
        prob_agr = f"{m['predicted_prob_agreement_pct']:.0f}" if m['predicted_prob_agreement_pct'] is not None else "N/A"
        all_val = f"{m['all_value_agreement_pct']:.0f}" if m['all_value_agreement_pct'] is not None else "N/A"
        
        row = (
            f"{provider:<12} {m['extractions_processed']:<6} "
            f"{m['avg_features_mentioned']:<12.2f} {m['avg_shap_feature_values_given']:<12.2f} {m['avg_protected_attrs_values_given']:<12.2f} {m['avg_other_feature_values_given']:<12.2f} {m['avg_protected_attrs_mentioned']:<14.2f} {m['avg_shap_values_given']:<12.2f} "
            f"{rank_1:<6} {rank_2:<6} {rank_3:<6} {rank_total:<6} {sign_1:<6} {sign_2:<6} {sign_3:<6} {sign_total:<6} "
            f"{shap_val:<14} {prob_ment:<12} {prob_agr:<12} {all_val:<12}"
        )
        print(row)
    
    print("=" * 280)


def save_results_to_csv(results_by_provider, dataset_name):
    """Save results to CSV file."""
    output_path = f"results/shap_metrics_{dataset_name}.csv"
    
    rows = []
    for provider, m in results_by_provider.items():
        rows.append({
            # Provider info
            "narrative_provider": m["provider"],
            "extractions_processed": m["extractions_processed"],
            
            # Feature extraction metrics
            "avg_features_mentioned": m["avg_features_mentioned"],
            "avg_protected_attrs_mentioned": m["avg_protected_attrs_mentioned"],
            "avg_shap_values_given": m["avg_shap_values_given"],
            
            # Feature values breakdown
            "avg_feature_values_given": m["avg_feature_values_given"],
            "avg_shap_feature_values_given": m["avg_shap_feature_values_given"],
            "avg_protected_attrs_values_given": m["avg_protected_attrs_values_given"],
            "avg_other_feature_values_given": m["avg_other_feature_values_given"],
            
            # Rank agreement
            "rank_1_agreement_%": m["rank_1_agreement_pct"],
            "rank_2_agreement_%": m["rank_2_agreement_pct"],
            "rank_3_agreement_%": m["rank_3_agreement_pct"],
            "rank_total_agreement_%": m["rank_total_agreement_pct"],
            
            # Sign agreement
            "sign_1_agreement_%": m["sign_1_agreement_pct"],
            "sign_2_agreement_%": m["sign_2_agreement_pct"],
            "sign_3_agreement_%": m["sign_3_agreement_pct"],
            "sign_total_agreement_%": m["sign_total_agreement_pct"],
            
            # Value agreements
            "shap_value_agreement_%": m["shap_value_agreement_pct"],
            "all_value_agreement_%": m["all_value_agreement_pct"],
            
            # Predicted probability
            "predicted_prob_mention_%": m["predicted_prob_mention_pct"],
            "predicted_prob_agreement_%": m["predicted_prob_agreement_pct"],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    return df


def compute_metrics_by_protected_attributes(dataset_name, prompt_type="shap"):
    """
    Compute metrics broken down by protected attribute values.
    This helps identify potential bias/fairness issues.
    For age: group into <=33 and >33
    """
    protected_attrs = DATASETS[dataset_name]["protected_attrs"]
    num_instances = DATASETS[dataset_name]["num_instances"]
    
    # Find all extraction files dynamically
    extraction_pattern = f"results/extractions/{dataset_name}/extractions/{prompt_type}/*/majority_voted/instance_*.json"
    extraction_files = glob.glob(extraction_pattern)
    
    if not extraction_files:
        print(f"[WARN] No extraction files found for {dataset_name}")
        return {}
    
    # Group instances by protected attribute values
    instances_by_protected_attr = {attr: defaultdict(list) for attr in protected_attrs}
    
    # First pass: collect all instances and their protected attribute values
    for instance_idx in range(num_instances):
        gt_path = f"results/ground_truth/{dataset_name}/instance_{instance_idx}.json"
        if os.path.exists(gt_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            
            # Extract protected attribute values
            for attr in protected_attrs:
                attr_feat = next((f for f in ground_truth.get("features", []) if f.get("name") == attr), None)
                if attr_feat:
                    attr_value = attr_feat.get("value")
                    
                    # Special handling for age: bin into <=33 and >33
                    if attr == "age":
                        binned_value = "<=33" if attr_value <= 33 else ">33"
                        instances_by_protected_attr[attr][binned_value].append(instance_idx)
                    else:
                        instances_by_protected_attr[attr][attr_value].append(instance_idx)
    
    # Second pass: calculate metrics for each protected attribute group
    results = {}
    
    for attr in protected_attrs:
        results[attr] = {}
        
        for attr_value in sorted(instances_by_protected_attr[attr].keys()):
            instance_indices = instances_by_protected_attr[attr][attr_value]
            
            # Calculate metrics for this group
            group_metrics = {
                "instances_count": 0,
                "features_mentioned_sum": 0,
                "feature_values_given_sum": 0,
                "protected_attrs_mentioned_sum": 0,
                "shap_values_given_sum": 0,
                "shap_feature_values_given_sum": 0,
                "protected_attrs_values_given_sum": 0,
                "other_feature_values_given_sum": 0,
                "rank_1_sum": 0,
                "rank_2_sum": 0,
                "rank_3_sum": 0,
                "rank_total_sum": 0,
                "rank_total_count": 0,
                "sign_1_sum": 0,
                "sign_2_sum": 0,
                "sign_3_sum": 0,
                "sign_total_sum": 0,
                "sign_total_count": 0,
                "shap_value_agreement_sum": 0,
                "shap_value_agreement_count": 0,
                "all_value_agreement_sum": 0,
                "all_value_agreement_count": 0,
            }
            
            # For each instance in this group, calculate metrics
            for instance_idx in instance_indices:
                gt_path = f"results/ground_truth/{dataset_name}/instance_{instance_idx}.json"
                ext_path = f"results/extractions/{dataset_name}/extractions/{prompt_type}/*/majority_voted/instance_{instance_idx}.json"
                
                # Find extraction file
                ext_files = glob.glob(ext_path)
                if not ext_files:
                    continue
                
                ext_path = ext_files[0]  # Use first match
                
                with open(gt_path) as f:
                    ground_truth = json.load(f)
                with open(ext_path) as f:
                    extraction = json.load(f)
                
                metrics = calculate_metrics(extraction, ground_truth, dataset_name)
                
                if metrics["valid"]:
                    group_metrics["instances_count"] += 1
                    
                    # Accumulate feature metrics
                    group_metrics["features_mentioned_sum"] += metrics["features_mentioned"]
                    group_metrics["feature_values_given_sum"] += metrics["feature_values_given"]
                    group_metrics["protected_attrs_mentioned_sum"] += metrics["protected_attrs_mentioned"]
                    group_metrics["shap_values_given_sum"] += metrics["shap_values_given"]
                    group_metrics["shap_feature_values_given_sum"] += metrics["shap_feature_values_given"]
                    group_metrics["protected_attrs_values_given_sum"] += metrics["protected_attrs_values_given"]
                    group_metrics["other_feature_values_given_sum"] += metrics["other_feature_values_given"]
                    
                    # Accumulate rank agreements
                    for i, rank_agree in enumerate(metrics["rank_agreements"]):
                        if rank_agree is not None:
                            group_metrics[f"rank_{i+1}_sum"] += rank_agree
                            group_metrics["rank_total_sum"] += rank_agree
                            group_metrics["rank_total_count"] += 1
                    
                    # Accumulate sign agreements
                    for i, sign_agree in enumerate(metrics["sign_agreements"]):
                        if sign_agree is not None:
                            group_metrics[f"sign_{i+1}_sum"] += sign_agree
                            group_metrics["sign_total_sum"] += sign_agree
                            group_metrics["sign_total_count"] += 1
                    
                    # Accumulate SHAP value agreements
                    for agreement in metrics["shap_value_agreement"]:
                        group_metrics["shap_value_agreement_sum"] += agreement
                        group_metrics["shap_value_agreement_count"] += 1
                    
                    # Accumulate all value agreements
                    for agreement in metrics["all_value_agreement"]:
                        group_metrics["all_value_agreement_sum"] += agreement
                        group_metrics["all_value_agreement_count"] += 1
            
            # Calculate averages and percentages
            results[attr][attr_value] = {
                "instances_count": group_metrics["instances_count"],
                "avg_features_mentioned": group_metrics["features_mentioned_sum"] / group_metrics["instances_count"] if group_metrics["instances_count"] > 0 else None,
                "avg_feature_values_given": group_metrics["feature_values_given_sum"] / group_metrics["instances_count"] if group_metrics["instances_count"] > 0 else None,
                "avg_protected_attrs_mentioned": group_metrics["protected_attrs_mentioned_sum"] / group_metrics["instances_count"] if group_metrics["instances_count"] > 0 else None,
                "avg_shap_values_given": group_metrics["shap_values_given_sum"] / group_metrics["instances_count"] if group_metrics["instances_count"] > 0 else None,
                "avg_shap_feature_values_given": group_metrics["shap_feature_values_given_sum"] / group_metrics["instances_count"] if group_metrics["instances_count"] > 0 else None,
                "avg_protected_attrs_values_given": group_metrics["protected_attrs_values_given_sum"] / group_metrics["instances_count"] if group_metrics["instances_count"] > 0 else None,
                "avg_other_feature_values_given": group_metrics["other_feature_values_given_sum"] / group_metrics["instances_count"] if group_metrics["instances_count"] > 0 else None,
                "rank_1_agreement_%": (group_metrics["rank_1_sum"] / 1 / group_metrics["instances_count"] * 100) if group_metrics["instances_count"] > 0 else None,
                "rank_2_agreement_%": (group_metrics["rank_2_sum"] / 1 / group_metrics["instances_count"] * 100) if group_metrics["instances_count"] > 0 else None,
                "rank_3_agreement_%": (group_metrics["rank_3_sum"] / 1 / group_metrics["instances_count"] * 100) if group_metrics["instances_count"] > 0 else None,
                "rank_total_agreement_%": (group_metrics["rank_total_sum"] / group_metrics["rank_total_count"] * 100) if group_metrics["rank_total_count"] > 0 else None,
                "sign_1_agreement_%": (group_metrics["sign_1_sum"] / 1 / group_metrics["instances_count"] * 100) if group_metrics["instances_count"] > 0 else None,
                "sign_2_agreement_%": (group_metrics["sign_2_sum"] / 1 / group_metrics["instances_count"] * 100) if group_metrics["instances_count"] > 0 else None,
                "sign_3_agreement_%": (group_metrics["sign_3_sum"] / 1 / group_metrics["instances_count"] * 100) if group_metrics["instances_count"] > 0 else None,
                "sign_total_agreement_%": (group_metrics["sign_total_sum"] / group_metrics["sign_total_count"] * 100) if group_metrics["sign_total_count"] > 0 else None,
                "shap_value_agreement_%": (group_metrics["shap_value_agreement_sum"] / group_metrics["shap_value_agreement_count"] * 100) if group_metrics["shap_value_agreement_count"] > 0 else None,
                "all_value_agreement_%": (group_metrics["all_value_agreement_sum"] / group_metrics["all_value_agreement_count"] * 100) if group_metrics["all_value_agreement_count"] > 0 else None,
            }
    
    return results


def print_protected_attributes_summary(results, dataset_name):
    """Print summary of metrics broken down by protected attributes."""
    print("\n" + "=" * 180)
    print(f"SHAP METRICS BY PROTECTED ATTRIBUTES - {dataset_name.upper()} DATASET")
    print("=" * 180)
    
    for attr in DATASETS[dataset_name]["protected_attrs"]:
        if attr not in results:
            continue
        
        print(f"\n{attr.upper()}:")
        print("-" * 180)
        print(f"{'Value':<15} {'N':<5} {'Features':<10} {'FeatVals':<10} {'ProtAttrs':<10} {'ShapVals':<10} "
              f"{'Rank1%':<8} {'Rank2%':<8} {'Rank3%':<8} {'RankTot%':<10} "
              f"{'Sign1%':<8} {'Sign2%':<8} {'Sign3%':<8} {'SignTot%':<10} "
              f"{'SHAP-Val%':<10} {'AllVal%':<10}")
        print("-" * 180)
        
        for attr_value in sorted(results[attr].keys()):
            metrics = results[attr][attr_value]
            print(f"{str(attr_value):<15} {metrics['instances_count']:<5} "
                  f"{metrics['avg_features_mentioned']:>9.2f} {metrics['avg_feature_values_given']:>9.2f} {metrics['avg_protected_attrs_mentioned']:>9.2f} {metrics['avg_shap_values_given']:>9.2f} "
                  f"{metrics['rank_1_agreement_%']:>7.1f}% {metrics['rank_2_agreement_%']:>7.1f}% {metrics['rank_3_agreement_%']:>7.1f}% {metrics['rank_total_agreement_%']:>9.1f}% "
                  f"{metrics['sign_1_agreement_%']:>7.1f}% {metrics['sign_2_agreement_%']:>7.1f}% {metrics['sign_3_agreement_%']:>7.1f}% {metrics['sign_total_agreement_%']:>9.1f}% "
                  f"{metrics['shap_value_agreement_%']:>9.1f}% {metrics['all_value_agreement_%']:>9.1f}%")


def save_protected_attributes_to_csv(results, dataset_name):
    """Save protected attributes metrics to CSV file."""
    output_path = f"results/shap_metrics_protected_attributes_{dataset_name}.csv"
    
    rows = []
    for attr in DATASETS[dataset_name]["protected_attrs"]:
        if attr not in results:
            continue
        
        for attr_value in sorted(results[attr].keys()):
            metrics = results[attr][attr_value]
            rows.append({
                # Attribute info
                "protected_attribute": attr,
                "attribute_value": str(attr_value),
                "instances_count": metrics["instances_count"],
                
                # Feature extraction metrics
                "avg_features_mentioned": metrics["avg_features_mentioned"],
                "avg_protected_attrs_mentioned": metrics["avg_protected_attrs_mentioned"],
                "avg_shap_values_given": metrics["avg_shap_values_given"],
                
                # Feature values breakdown
                "avg_feature_values_given": metrics["avg_feature_values_given"],
                "avg_shap_feature_values_given": metrics["avg_shap_feature_values_given"],
                "avg_protected_attrs_values_given": metrics["avg_protected_attrs_values_given"],
                "avg_other_feature_values_given": metrics["avg_other_feature_values_given"],
                
                # Rank agreement
                "rank_1_agreement_%": metrics["rank_1_agreement_%"],
                "rank_2_agreement_%": metrics["rank_2_agreement_%"],
                "rank_3_agreement_%": metrics["rank_3_agreement_%"],
                "rank_total_agreement_%": metrics["rank_total_agreement_%"],
                
                # Sign agreement
                "sign_1_agreement_%": metrics["sign_1_agreement_%"],
                "sign_2_agreement_%": metrics["sign_2_agreement_%"],
                "sign_3_agreement_%": metrics["sign_3_agreement_%"],
                "sign_total_agreement_%": metrics["sign_total_agreement_%"],
                
                # Value agreements
                "shap_value_agreement_%": metrics["shap_value_agreement_%"],
                "all_value_agreement_%": metrics["all_value_agreement_%"],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Protected attributes metrics saved to: {output_path}")
    return df


def main():
    print("\n" + "=" * 200)
    print("CALCULATING SHAP NARRATIVE EXTRACTION METRICS")
    print("=" * 200)
    
    for dataset_name in ["credit", "law"]:
        print(f"\n[INFO] Processing {dataset_name.upper()} dataset...")
        
        results_by_provider = aggregate_metrics_by_provider(dataset_name, prompt_type="shap")
        
        if results_by_provider:
            print_summary(results_by_provider, dataset_name)
            save_results_to_csv(results_by_provider, dataset_name)
            
            # Print protected attributes breakdown
            print("\n[INFO] Computing metrics by protected attributes...")
            protected_attr_results = compute_metrics_by_protected_attributes(dataset_name, prompt_type="shap")
            if protected_attr_results:
                print_protected_attributes_summary(protected_attr_results, dataset_name)
                save_protected_attributes_to_csv(protected_attr_results, dataset_name)
        else:
            print(f"[WARN] No extraction data found for {dataset_name}")


if __name__ == "__main__":
    main()
