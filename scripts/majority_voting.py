"""
Majority Voting Script: Combine multiple LLM extractions via majority voting.

This script:
1. Loads extractions from all extractor LLMs (grok, deepseek, gemini, openai, claude, mistral)
2. Performs majority voting on ranks, signs, values, and feature names
3. Saves the final voted extraction to: 
   results/extractions/{dataset}/extractions/{prompt_type}/{narrative_provider}/majority_voted/instance_{idx}.json

Run:
    python scripts/majority_voting.py
"""

import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime
import glob

# Add parent path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "credit"  # "credit" or "law"
PROMPT_TYPE = "shap"  # "shap" or "cf"
NARRATIVE_PROVIDERS = ["gemini", "grok", "deepseek", "mistral", "openai", "claude"]
EXTRACTOR_PROVIDERS = ["gemini", "grok", "deepseek", "mistral", "openai", "claude"]

DATASETS = {
    "credit": {"num_instances": 34},
    "law": {"num_instances": 308}
}

# ============================================================================
# DISAGREEMENT TRACKING
# ============================================================================

# Global statistics for disagreement analysis
DISAGREEMENT_STATS = {
    "rank_1_disagreements": 0,
    "rank_1_total": 0,
    "rank_2_disagreements": 0,
    "rank_2_total": 0,
    "rank_3_disagreements": 0,
    "rank_3_total": 0,
    "disagreement_instances": defaultdict(lambda: defaultdict(list)),  # {rank: {extractor: [instances]}}
}


def track_rank_disagreement(rank, voted_name, extractor_features, extractor_names):
    """Track when extractors disagreed on a rank position."""
    for extractor_idx, extractor_name in enumerate(extractor_names):
        if extractor_idx < len(extractor_features):
            features_at_rank = extractor_features[extractor_idx].get(rank, [])
            if features_at_rank:
                extractor_name_at_rank = features_at_rank[0].get("name") if features_at_rank else None
                if extractor_name_at_rank and extractor_name_at_rank != voted_name:
                    DISAGREEMENT_STATS["disagreement_instances"][f"rank_{rank}"][extractor_name].append(extractor_name_at_rank)





def load_extraction(dataset_name, instance_idx, narrative_provider, extractor_provider, prompt_type):
    """Load a single extraction JSON."""
    path = f"results/extractions/{dataset_name}/extractions/{prompt_type}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def majority_vote_numeric(values, default=None):
    """Majority vote on numeric values. Returns most common value."""
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return default
    
    # Count occurrences
    counter = Counter(valid_values)
    most_common = counter.most_common(1)[0][0]
    return most_common


def majority_vote_string(values, default=None):
    """Majority vote on string values. Returns most common value."""
    valid_values = [v for v in values if v is not None and v != "NaN"]
    if not valid_values:
        return default
    
    counter = Counter(valid_values)
    most_common = counter.most_common(1)[0][0]
    return most_common


def majority_vote_predicted_probability(extractions):
    """Average predicted probability across all extractions."""
    probabilities = []
    for ext in extractions:
        if ext and "predicted_probability" in ext:
            prob = ext["predicted_probability"]
            if isinstance(prob, (int, float)):
                probabilities.append(prob)
            elif isinstance(prob, str) and prob not in ["NaN", "None"]:
                try:
                    probabilities.append(float(prob))
                except (ValueError, TypeError):
                    pass
    
    if probabilities:
        # Return average
        return round(sum(probabilities) / len(probabilities), 2)
    return None


def majority_vote_most_important_features(extractions, num_features=3, track_disagreements=False):
    """
    Majority vote on most important features.
    
    Strategy:
    1. For each rank position (1,2,3), vote on which feature name appears most
    2. For that feature, vote on sign and value
    3. Build final list sorted by rank
    
    Returns:
        voted_features: list of voted features
        disagreement_info: dict with disagreement details (if track_disagreements=True)
    """
    
    # Collect all most_important_features from all extractions
    all_features_by_rank = defaultdict(lambda: defaultdict(list))  # rank -> {extractor_idx: [features]}
    extractor_names = []
    
    for ext_idx, ext in enumerate(extractions):
        extractor_names.append(f"extractor_{ext_idx}")
        
        if not ext or "most_important_features" not in ext:
            continue
        
        for feat in ext["most_important_features"]:
            rank = feat.get("rank")
            name = feat.get("name")
            sign = feat.get("sign")
            value = feat.get("value")
            
            if rank and name:
                all_features_by_rank[rank][ext_idx].append({
                    "name": name,
                    "sign": sign,
                    "value": value
                })
    
    # Vote for each rank position
    voted_features = []
    disagreement_info = {"rank_1": {}, "rank_2": {}, "rank_3": {}}
    
    for rank in sorted(all_features_by_rank.keys())[:num_features]:
        features_at_rank_by_extractor = all_features_by_rank[rank]
        
        # Collect all feature names at this rank from all extractors
        names = []
        for ext_idx in range(len(extractions)):
            if ext_idx in features_at_rank_by_extractor:
                features = features_at_rank_by_extractor[ext_idx]
                if features:
                    names.append(features[0]["name"])
        
        if not names:
            continue
        
        # Vote on feature name
        voted_name = majority_vote_string(names)
        
        # Track disagreements
        if track_disagreements:
            names_counter = Counter(names)
            total_votes = len(names)
            max_votes = names_counter[voted_name]
            
            disagreement_info[f"rank_{rank}"] = {
                "voted_name": voted_name,
                "votes": max_votes,
                "total": total_votes,
                "unanimous": (max_votes == total_votes),
                "disagreeing_extractors": []
            }
            
            # Find which extractors disagreed
            for ext_idx, ext_name in enumerate(extractor_names):
                if ext_idx < len(extractions) and ext_idx in features_at_rank_by_extractor:
                    features = features_at_rank_by_extractor[ext_idx]
                    if features and features[0]["name"] != voted_name:
                        disagreement_info[f"rank_{rank}"]["disagreeing_extractors"].append({
                            "extractor": ext_name,
                            "voted_for": features[0]["name"]
                        })
        
        if voted_name:
            # For this feature name, collect all instances of it and vote on sign/value
            instances_of_feature = []
            for ext_idx in range(len(extractions)):
                if ext_idx in features_at_rank_by_extractor:
                    features = features_at_rank_by_extractor[ext_idx]
                    for f in features:
                        if f["name"] == voted_name:
                            instances_of_feature.append(f)
                            break
            
            signs = [f["sign"] for f in instances_of_feature if f["sign"] is not None]
            values = [f["value"] for f in instances_of_feature if f["value"] is not None]
            
            voted_sign = majority_vote_numeric(signs, default=1)
            voted_value = majority_vote_numeric(values, default="NaN") if values else "NaN"
            
            voted_features.append({
                "rank": rank,
                "name": voted_name,
                "sign": voted_sign,
                "value": voted_value
            })
    
    if track_disagreements:
        return voted_features, disagreement_info
    return voted_features


def majority_vote_features_array(extractions):
    """
    Majority vote on features array.
    
    Strategy:
    1. For each feature name that appears, vote on whether it was mentioned
    2. For features that are mentioned (majority), vote on the value
    """
    
    all_features_dict = defaultdict(lambda: {"mentioned": [], "values": []})
    
    for ext in extractions:
        if not ext or "features" not in ext:
            continue
        
        for feat in ext["features"]:
            name = feat.get("name")
            mentioned = feat.get("mentioned", 0)
            value = feat.get("value")
            
            if name:
                # Convert mentioned to int in case it's a string
                mentioned_int = int(mentioned) if isinstance(mentioned, str) else mentioned
                all_features_dict[name]["mentioned"].append(mentioned_int)
                if value is not None:
                    all_features_dict[name]["values"].append(value)
    
    # Vote for each feature
    voted_features = []
    for name in sorted(all_features_dict.keys()):
        feature_data = all_features_dict[name]
        
        # Vote on whether mentioned (majority vote on binary 0/1)
        mentioned_votes = feature_data["mentioned"]
        voted_mentioned = 1 if sum(mentioned_votes) > len(mentioned_votes) / 2 else 0
        
        # Vote on value
        values = feature_data["values"]
        if values:
            # Try to vote on numeric values first
            numeric_values = []
            non_numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                elif isinstance(v, str):
                    if v not in ["NaN", "None"]:
                        try:
                            numeric_values.append(float(v))
                        except (ValueError, TypeError):
                            non_numeric_values.append(v)
            
            # If we have numeric values, use median/mode of those
            if numeric_values:
                voted_value = majority_vote_numeric(numeric_values, default="NaN")
            elif non_numeric_values:
                voted_value = majority_vote_string(non_numeric_values, default="NaN")
            else:
                voted_value = "NaN"
        else:
            voted_value = "NaN"
        
        voted_features.append({
            "name": name,
            "mentioned": voted_mentioned,
            "value": voted_value
        })
    
    return voted_features


def majority_vote_extraction(extractions):
    """
    Combine multiple extractions via majority voting.
    
    Args:
        extractions: List of extraction dicts (can include None values for missing files)
    
    Returns:
        Tuple: (voted extraction dict, disagreement_info dict)
    """
    
    # Filter out None values
    valid_extractions = [e for e in extractions if e is not None]
    
    if not valid_extractions:
        return None, None
    
    # Get most important features with disagreement tracking
    features_result = majority_vote_most_important_features(valid_extractions, track_disagreements=True)
    
    if isinstance(features_result, tuple):
        voted_features, disagreement_info = features_result
    else:
        voted_features = features_result
        disagreement_info = {}
    
    # Voted result
    result = {
        "predicted_probability": majority_vote_predicted_probability(valid_extractions),
        "most_important_features": voted_features,
        "features": majority_vote_features_array(valid_extractions)
    }
    
    return result, disagreement_info


def process_instance(dataset_name, instance_idx, narrative_provider, extractor_providers, prompt_type):
    """Process a single instance through majority voting and track disagreements."""
    
    # Load all extractions for this instance
    extractions = []
    for extractor in extractor_providers:
        ext = load_extraction(dataset_name, instance_idx, narrative_provider, extractor, prompt_type)
        extractions.append(ext)
    
    # Perform majority voting
    result = majority_vote_extraction(extractions)
    
    if isinstance(result, tuple):
        voted, disagreement_info = result
    else:
        voted = result
        disagreement_info = None
    
    if voted:
        # Track disagreements
        if disagreement_info:
            for rank in ["rank_1", "rank_2", "rank_3"]:
                if rank in disagreement_info and disagreement_info[rank]:
                    rank_num = int(rank.split("_")[1])
                    is_unanimous = disagreement_info[rank].get("unanimous", False)
                    
                    if not is_unanimous:
                        DISAGREEMENT_STATS[f"{rank}_disagreements"] += 1
                    
                    DISAGREEMENT_STATS[f"{rank}_total"] += 1
                    
                    # Track which extractors disagreed
                    for disagreer in disagreement_info[rank].get("disagreeing_extractors", []):
                        extractor_name = disagreer["extractor"]
                        DISAGREEMENT_STATS["disagreement_instances"][rank][extractor_name].append(
                            f"inst_{instance_idx}_{disagreer['voted_for']}"
                        )
        
        # Save to majority_voted directory
        output_dir = f"results/extractions/{dataset_name}/extractions/{prompt_type}/{narrative_provider}/majority_voted"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/instance_{instance_idx}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(voted, f, indent=2)
        
        return True, output_file
    else:
        return False, "No valid extractions found"


def generate_disagreement_report():
    """Generate a report on extractor disagreements."""
    
    print("\n" + "=" * 100)
    print("DISAGREEMENT ANALYSIS - Extractor Agreement Summary")
    print("=" * 100)
    
    # Summary statistics
    print("\nRank-wise Disagreement Summary:")
    print("-" * 100)
    
    for rank in [1, 2, 3]:
        rank_key = f"rank_{rank}"
        disagreements = DISAGREEMENT_STATS[f"{rank_key}_disagreements"]
        total = DISAGREEMENT_STATS[f"{rank_key}_total"]
        
        if total > 0:
            pct = 100 * disagreements / total
            agreement_pct = 100 - pct
            print(f"  Rank {rank}: {disagreements}/{total} disagreements ({pct:.1f}%) | Agreement: {agreement_pct:.1f}%")
        else:
            print(f"  Rank {rank}: No data")
    
    # Per-extractor disagreement counts
    print("\n" + "-" * 100)
    print("Extractor-wise Disagreement Frequency:")
    print("-" * 100)
    
    extractor_disagreement_count = defaultdict(int)
    extractor_instances = defaultdict(list)
    
    for rank, extractors_dict in DISAGREEMENT_STATS["disagreement_instances"].items():
        for extractor_name, instances in extractors_dict.items():
            extractor_disagreement_count[extractor_name] += len(instances)
            extractor_instances[extractor_name].extend(instances)
    
    if extractor_disagreement_count:
        # Sort by disagreement count descending
        sorted_extractors = sorted(extractor_disagreement_count.items(), key=lambda x: x[1], reverse=True)
        
        for extractor_name, count in sorted_extractors:
            print(f"  {extractor_name:15}: {count:3} disagreements")
            # Show top 5 instances for this extractor
            top_instances = extractor_instances[extractor_name][:5]
            if top_instances:
                print(f"    Examples: {', '.join(top_instances[:3])}")
    else:
        print("  No disagreements found!")
    
    print("\n" + "=" * 100)


def run_majority_voting():
    """Main majority voting pipeline."""
    
    print("\n" + "=" * 100)
    print(f"MAJORITY VOTING PIPELINE")
    print("=" * 100)
    print(f"Dataset: {DATASET.upper()}")
    print(f"Prompt type: {PROMPT_TYPE.upper()}")
    print(f"Narrative providers: {', '.join(NARRATIVE_PROVIDERS)}")
    print(f"Extractor LLMs: {', '.join(EXTRACTOR_PROVIDERS)}")
    print(f"Instances: {DATASETS[DATASET]['num_instances']}")
    print("=" * 100)
    
    start_time = datetime.now()
    
    total_success = 0
    total_failed = 0
    
    total_extractions = len(NARRATIVE_PROVIDERS) * DATASETS[DATASET]['num_instances']
    count = 0
    
    for narrative_provider in NARRATIVE_PROVIDERS:
        for instance_idx in range(DATASETS[DATASET]['num_instances']):
            count += 1
            
            # Progress
            if count % 10 == 1:
                pct = 100 * count // total_extractions
                elapsed = (datetime.now() - start_time).total_seconds()
                if count > 1:
                    rate = elapsed / (count - 1)
                    remaining = (total_extractions - count) * rate
                    eta_str = f" - ETA: {int(remaining//60)}m {int(remaining%60)}s"
                else:
                    eta_str = ""
                print(f"Progress: {count}/{total_extractions} ({pct}%){eta_str}")
            
            success, result = process_instance(
                dataset_name=DATASET,
                instance_idx=instance_idx,
                narrative_provider=narrative_provider,
                extractor_providers=EXTRACTOR_PROVIDERS,
                prompt_type=PROMPT_TYPE
            )
            
            if success:
                total_success += 1
            else:
                total_failed += 1
    
    # Summary
    elapsed = datetime.now() - start_time
    print("\n" + "=" * 100)
    print(f"MAJORITY VOTING COMPLETE")
    print(f"Total: {total_success + total_failed}")
    print(f"  ✅ Voted: {total_success}")
    print(f"  ❌ Failed: {total_failed}")
    print(f"Time: {int(elapsed.total_seconds()//60)}m {int(elapsed.total_seconds()%60)}s")
    print("=" * 100)
    print(f"\n📁 Voted extractions saved to:")
    print(f"   results/extractions/{DATASET}/extractions/{PROMPT_TYPE}/<narrative_provider>/majority_voted/")
    print("=" * 100)
    
    # Print disagreement analysis
    generate_disagreement_report()


if __name__ == "__main__":
    run_majority_voting()
