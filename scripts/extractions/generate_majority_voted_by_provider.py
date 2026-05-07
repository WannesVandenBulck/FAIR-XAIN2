"""
Generate majority voted extractions for each narrative provider.

For each of the 6 narrative providers (openai, deepseek, grok, claude, gemini, mistral):
  - Load extractions from all 3 available extractors (openai, deepseek, grok)
  - Majority vote them
  - Save to results/extractions/majority/{narrative_provider}/instance_{idx}.json

Run:
    python scripts/generate_majority_voted_by_provider.py
"""

import os
import json
from collections import Counter, defaultdict
from pathlib import Path

DATASET = "credit"
PROMPT_TYPE = "shap"
NARRATIVE_PROVIDERS = ["openai", "gemini", "grok", "deepseek", "mistral", "claude"]  # All 6 narrative providers
EXTRACTOR_PROVIDERS = ["openai", "deepseek", "grok"]  # Only these 3 have extractions
NUM_INSTANCES = 34


def load_extraction(narrative_provider, extractor_provider, instance_idx):
    """Load extraction JSON."""
    path = f"results/extractions/{DATASET}/extractions/{PROMPT_TYPE}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def normalize_numeric_value(value):
    """Normalize numeric values to integers for consistent voting.
    
    Converts string/float representations to integers.
    Keeps "NaN" as "NaN".
    """
    if value == "NaN" or value is None:
        return "NaN"
    try:
        # Convert to float first (handles "5", 5, 5.0), then to int
        return int(float(value))
    except (ValueError, TypeError):
        # Non-numeric values stay as-is
        return value


def majority_vote_string(values):
    """Majority vote on string values."""
    valid_values = [v for v in values if v is not None and v != "NaN"]
    if not valid_values:
        return "NaN"
    counter = Counter(valid_values)
    return counter.most_common(1)[0][0]


def majority_vote_numeric(values):
    """Majority vote on numeric values."""
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None
    counter = Counter(valid_values)
    return counter.most_common(1)[0][0]


def majority_vote_most_important_features(extractions):
    """Majority vote on most important features (ranks 1, 2, 3).
    
    Normalizes numeric values BEFORE voting to handle type inconsistencies.
    - Numeric values are converted to integers
    - "NaN" stays as "NaN"
    """
    # Collect features by rank
    features_by_rank = defaultdict(lambda: {"names": [], "signs": [], "values": []})
    
    for ext in extractions:
        if not ext or "most_important_features" not in ext:
            continue
        
        for feat in ext.get("most_important_features", []):
            rank = feat.get("rank")
            if rank in [1, 2, 3]:
                features_by_rank[rank]["names"].append(feat.get("name"))
                features_by_rank[rank]["signs"].append(feat.get("sign"))
                # Normalize value BEFORE adding to vote collection
                normalized_val = normalize_numeric_value(feat.get("value"))
                features_by_rank[rank]["values"].append(normalized_val)
    
    # Majority vote for each rank
    voted_features = []
    for rank in [1, 2, 3]:
        if rank in features_by_rank:
            voted_name = majority_vote_string(features_by_rank[rank]["names"])
            voted_sign = majority_vote_numeric(features_by_rank[rank]["signs"])
            # Values are already normalized, vote on them directly
            voted_value = majority_vote_string(features_by_rank[rank]["values"])
            
            voted_features.append({
                "rank": rank,
                "name": voted_name,
                "sign": voted_sign,
                "value": voted_value
            })
    
    return voted_features


def majority_vote_features(extractions):
    """Majority vote on all features.
    
    Normalizes numeric values BEFORE voting to handle type inconsistencies.
    - Numeric values are converted to integers
    - "NaN" stays as "NaN"
    """
    features_dict = defaultdict(lambda: {"mentioned": [], "values": []})
    
    for ext in extractions:
        if not ext or "features" not in ext:
            continue
        
        for feat in ext.get("features", []):
            name = feat.get("name")
            if name:
                features_dict[name]["mentioned"].append(feat.get("mentioned", 0))
                # Normalize value BEFORE adding to vote collection
                normalized_val = normalize_numeric_value(feat.get("value"))
                features_dict[name]["values"].append(normalized_val)
    
    # Majority vote for each feature
    voted_features = []
    for name in sorted(features_dict.keys()):
        voted_mentioned = majority_vote_numeric(features_dict[name]["mentioned"])
        # Values are already normalized, vote on them directly
        voted_value = majority_vote_string(features_dict[name]["values"])
        
        voted_features.append({
            "name": name,
            "mentioned": voted_mentioned if voted_mentioned is not None else 0,
            "value": voted_value
        })
    
    return voted_features


def majority_vote_instance(narrative_provider, instance_idx):
    """Majority vote extractions from all extractors for a given instance."""
    # Load all extractor versions
    extractions = []
    for extractor in EXTRACTOR_PROVIDERS:
        ext = load_extraction(narrative_provider, extractor, instance_idx)
        if ext:
            extractions.append(ext)
    
    if not extractions:
        print(f"  [WARN] No extractions found for {narrative_provider} instance {instance_idx}")
        return None
    
    # Majority vote on predicted probability
    probs = []
    for ext in extractions:
        prob = ext.get("predicted_probability")
        if prob is not None:
            if isinstance(prob, str):
                try:
                    probs.append(float(prob))
                except:
                    pass
            else:
                probs.append(float(prob))
    
    voted_prob = round(sum(probs) / len(probs), 2) if probs else None
    
    # Create voted extraction
    voted_extraction = {
        "predicted_probability": voted_prob,
        "most_important_features": majority_vote_most_important_features(extractions),
        "features": majority_vote_features(extractions)
    }
    
    return voted_extraction


def save_extraction(narrative_provider, instance_idx, voted_extraction):
    """Save voted extraction JSON."""
    output_dir = f"results/extractions/majority/{narrative_provider}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/instance_{instance_idx}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(voted_extraction, f, indent=2)


def main():
    print("\n" + "="*100)
    print("GENERATING MAJORITY VOTED EXTRACTIONS BY PROVIDER")
    print("="*100)
    
    for narrative_provider in NARRATIVE_PROVIDERS:
        print(f"\n[{narrative_provider.upper()}]")
        
        for instance_idx in range(NUM_INSTANCES):
            voted = majority_vote_instance(narrative_provider, instance_idx)
            if voted:
                save_extraction(narrative_provider, instance_idx, voted)
                if (instance_idx + 1) % 10 == 0:
                    print(f"  Processed instances 0-{instance_idx}")
        
        print(f"  [OK] Completed {narrative_provider}")
    
    print("\n" + "="*100)
    print("DONE")
    print("="*100)


if __name__ == "__main__":
    main()
