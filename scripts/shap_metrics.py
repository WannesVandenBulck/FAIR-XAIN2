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
        "shap_values_given": 0,            # NEW: count of SHAP features with values given
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
    
    # Create maps for easy lookup by rank
    gt_by_rank = {feat.get("rank"): feat for feat in ground_truth_shap}
    
    for rank_pos in range(1, 4):  # Ranks 1, 2, 3
        ext_feat = next((f for f in extraction_shap if f.get("rank") == rank_pos), None)
        gt_feat = gt_by_rank.get(rank_pos)
        
        if ext_feat and gt_feat:
            # Rank agreement - does the feature name match?
            if ext_feat.get("name") == gt_feat.get("name"):
                metrics["rank_agreements"][rank_pos - 1] = 1
            else:
                metrics["rank_agreements"][rank_pos - 1] = 0
            
            # Sign agreement - does the sign match?
            ext_sign = ext_feat.get("sign")
            gt_sign = gt_feat.get("sign")
            if ext_sign is not None and gt_sign is not None:
                if ext_sign == gt_sign:
                    metrics["sign_agreements"][rank_pos - 1] = 1
                else:
                    metrics["sign_agreements"][rank_pos - 1] = 0
            
            # Value agreement for SHAP features
            ext_value = ext_feat.get("value")
            gt_value = gt_feat.get("value")
            if ext_value and ext_value != "NaN" and gt_value and gt_value != "NaN":
                try:
                    ext_val_num = float(ext_value)
                    gt_val_num = float(gt_value)
                    metrics["shap_value_agreement"].append(abs(ext_val_num - gt_val_num) < 0.1)
                except (ValueError, TypeError):
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
    # Find all extraction files dynamically
    extraction_pattern = f"results/extractions/{dataset_name}/extractions/{prompt_type}/*/*/instance_*.json"
    extraction_files = glob.glob(extraction_pattern)
    
    if not extraction_files:
        print(f"⚠️  No extraction files found for {dataset_name}")
        return {}
    
    print(f"Found {len(extraction_files)} extraction files for {dataset_name}")
    
    # Parse extraction files to find which providers/narratives/instances exist
    extraction_map = defaultdict(list)  # narrative_provider -> list of (extractor_provider, instance_idx)
    
    for file_path in extraction_files:
        # Extract info from path: results/extractions/{dataset}/extractions/{prompt}/{narrative}/{extractor}/instance_{idx}.json
        parts = file_path.split(os.sep)
        try:
            narrative_provider = parts[-3]
            extractor_provider = parts[-2]
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
            
            # (d) rank agreement per rank
            "rank_1_agreement_sum": 0,
            "rank_1_agreement_count": 0,
            "rank_2_agreement_sum": 0,
            "rank_2_agreement_count": 0,
            "rank_3_agreement_sum": 0,
            "rank_3_agreement_count": 0,
            
            # (e) sign agreement per rank
            "sign_1_agreement_sum": 0,
            "sign_1_agreement_count": 0,
            "sign_2_agreement_sum": 0,
            "sign_2_agreement_count": 0,
            "sign_3_agreement_sum": 0,
            "sign_3_agreement_count": 0,
            
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
                
                # (d) rank agreements per position
                for rank_pos in range(3):
                    if metrics["rank_agreements"][rank_pos] is not None:
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
                
                # (d) rank agreement %
                "rank_1_agreement_pct": (provider_metrics["rank_1_agreement_sum"] / provider_metrics["rank_1_agreement_count"] * 100) if provider_metrics["rank_1_agreement_count"] > 0 else None,
                "rank_2_agreement_pct": (provider_metrics["rank_2_agreement_sum"] / provider_metrics["rank_2_agreement_count"] * 100) if provider_metrics["rank_2_agreement_count"] > 0 else None,
                "rank_3_agreement_pct": (provider_metrics["rank_3_agreement_sum"] / provider_metrics["rank_3_agreement_count"] * 100) if provider_metrics["rank_3_agreement_count"] > 0 else None,
                
                # (e) sign agreement %
                "sign_1_agreement_pct": (provider_metrics["sign_1_agreement_sum"] / provider_metrics["sign_1_agreement_count"] * 100) if provider_metrics["sign_1_agreement_count"] > 0 else None,
                "sign_2_agreement_pct": (provider_metrics["sign_2_agreement_sum"] / provider_metrics["sign_2_agreement_count"] * 100) if provider_metrics["sign_2_agreement_count"] > 0 else None,
                "sign_3_agreement_pct": (provider_metrics["sign_3_agreement_sum"] / provider_metrics["sign_3_agreement_count"] * 100) if provider_metrics["sign_3_agreement_count"] > 0 else None,
                
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
    print("\n" + "=" * 220)
    print(f"SHAP NARRATIVE EXTRACTION METRICS - {dataset_name.upper()} DATASET (by Narrative Provider)")
    print("=" * 220)
    
    # Print header line 1: Main categories
    header1 = (
        f"{'Provider':<12} {'Extr.':<6} "
        f"{'(a) Features':<12} {'(b) Values':<12} {'(c) Prot.Attrs':<14} {'Avg SHAP Val':<12} "
        f"{'(d) Rank Agreement %':<20} {'(e) Sign Agreement %':<20} "
        f"{'(f) SHAP Val %':<14} {'(g) Prob Ment %':<12} {'(g) Prob Agr %':<12} {'(h) Val Agr %':<12}"
    )
    print(header1)
    
    # Print header line 2: Sub-ranks
    header2 = (
        f"{'':12} {'':6} "
        f"{'Avg':<12} {'Avg':<12} {'Avg':<14} {'Avg':<12} "
        f"{'R1%':<6} {'R2%':<6} {'R3%':<6} {'S1%':<6} {'S2%':<6} {'S3%':<6} "
        f"{'%':<14} {'%':<12} {'%':<12} {'%':<12}"
    )
    print(header2)
    print("-" * 220)
    
    for provider in sorted(results_by_provider.keys()):
        m = results_by_provider[provider]
        
        # Format percentages
        rank_1 = f"{m['rank_1_agreement_pct']:.0f}" if m['rank_1_agreement_pct'] is not None else "N/A"
        rank_2 = f"{m['rank_2_agreement_pct']:.0f}" if m['rank_2_agreement_pct'] is not None else "N/A"
        rank_3 = f"{m['rank_3_agreement_pct']:.0f}" if m['rank_3_agreement_pct'] is not None else "N/A"
        
        sign_1 = f"{m['sign_1_agreement_pct']:.0f}" if m['sign_1_agreement_pct'] is not None else "N/A"
        sign_2 = f"{m['sign_2_agreement_pct']:.0f}" if m['sign_2_agreement_pct'] is not None else "N/A"
        sign_3 = f"{m['sign_3_agreement_pct']:.0f}" if m['sign_3_agreement_pct'] is not None else "N/A"
        
        shap_val = f"{m['shap_value_agreement_pct']:.0f}" if m['shap_value_agreement_pct'] is not None else "N/A"
        prob_ment = f"{m['predicted_prob_mention_pct']:.0f}"
        prob_agr = f"{m['predicted_prob_agreement_pct']:.0f}" if m['predicted_prob_agreement_pct'] is not None else "N/A"
        all_val = f"{m['all_value_agreement_pct']:.0f}" if m['all_value_agreement_pct'] is not None else "N/A"
        
        row = (
            f"{provider:<12} {m['extractions_processed']:<6} "
            f"{m['avg_features_mentioned']:<12.2f} {m['avg_feature_values_given']:<12.2f} {m['avg_protected_attrs_mentioned']:<14.2f} {m['avg_shap_values_given']:<12.2f} "
            f"{rank_1:<6} {rank_2:<6} {rank_3:<6} {sign_1:<6} {sign_2:<6} {sign_3:<6} "
            f"{shap_val:<14} {prob_ment:<12} {prob_agr:<12} {all_val:<12}"
        )
        print(row)
    
    print("=" * 220)


def save_results_to_csv(results_by_provider, dataset_name):
    """Save results to CSV file."""
    output_path = f"results/shap_metrics_{dataset_name}.csv"
    
    rows = []
    for provider, m in results_by_provider.items():
        rows.append({
            "narrative_provider": m["provider"],
            "extractions_processed": m["extractions_processed"],
            "avg_features_mentioned": m["avg_features_mentioned"],
            "avg_feature_values_given": m["avg_feature_values_given"],
            "avg_protected_attrs_mentioned": m["avg_protected_attrs_mentioned"],
            "avg_shap_values_given": m["avg_shap_values_given"],
            "rank_1_agreement_%": m["rank_1_agreement_pct"],
            "rank_2_agreement_%": m["rank_2_agreement_pct"],
            "rank_3_agreement_%": m["rank_3_agreement_pct"],
            "sign_1_agreement_%": m["sign_1_agreement_pct"],
            "sign_2_agreement_%": m["sign_2_agreement_pct"],
            "sign_3_agreement_%": m["sign_3_agreement_pct"],
            "shap_value_agreement_%": m["shap_value_agreement_pct"],
            "predicted_prob_mention_%": m["predicted_prob_mention_pct"],
            "predicted_prob_agreement_%": m["predicted_prob_agreement_pct"],
            "all_value_agreement_%": m["all_value_agreement_pct"],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    return df


def main():
    print("\n" + "=" * 200)
    print("CALCULATING SHAP NARRATIVE EXTRACTION METRICS")
    print("=" * 200)
    
    for dataset_name in ["credit", "law"]:
        print(f"\n📊 Processing {dataset_name.upper()} dataset...")
        
        results_by_provider = aggregate_metrics_by_provider(dataset_name, prompt_type="shap")
        
        if results_by_provider:
            print_summary(results_by_provider, dataset_name)
            save_results_to_csv(results_by_provider, dataset_name)
        else:
            print(f"⚠️  No extraction data found for {dataset_name}")


if __name__ == "__main__":
    main()
