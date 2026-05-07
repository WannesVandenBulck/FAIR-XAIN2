"""
Fairness Evaluation Framework for Narrative Faithfulness

This module provides tools to evaluate fairness in LLM narrative generation by:
1. Loading negatively predicted instances
2. Generating narratives with modified protected attributes (8 combinations per instance)
3. Extracting from these narratives using LLM extractors
4. Computing SHAP metrics grouped by attribute combinations
5. Analyzing fairness issues across protected attribute groups

Structure:
- results/fairness_eval/{batch_name}/narratives/{provider}/instance_{idx}_batch_{batch_id}.json
- results/fairness_eval/{batch_name}/extractions/{extractor}/{provider}/instance_{idx}_batch_{batch_id}.json
- results/fairness_eval/{batch_name}/metrics/{batch_name}_metrics_by_batch.csv

Usage:
    # 1. Get negative instances
    from fairness_evaluation import get_negatively_predicted_instances
    neg_instances = get_negatively_predicted_instances("credit", threshold=0.5)
    
    # 2. Generate narratives for fairness evaluation
    from fairness_evaluation import generate_fairness_narratives
    generate_fairness_narratives("credit", neg_instances[:50], 
                                providers=["openai", "grok", "deepseek", "mistral"], 
                                batch_name="fairness_v1")
    
    # 3. Extract from narratives
    from fairness_evaluation import extract_from_fairness_narratives
    extract_from_fairness_narratives("credit", 
                                    extractors=["openai", "grok", "deepseek"],
                                    batch_name="fairness_v1")
    
    # 4. Calculate metrics by attribute combination
    from fairness_evaluation import compute_metrics_by_attribute_combinations
    metrics_df = compute_metrics_by_attribute_combinations("credit", batch_name="fairness_v1")
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import itertools
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_tools.prompts import prompt_credit, prompt_law


# ===== CONFIGURATION =====

DATASETS = {
    "credit": {
        "adverse_file": "datasets_prep/data/credit_dataset/credit_adverse.csv",
        "shap_file": "datasets_prep/data/credit_dataset/credit_shap.csv",
        "protected_attrs": ["sex", "age", "foreign_worker"],
        "attribute_bins": {
            "sex": ["male", "female"],
            "foreign_worker": ["yes", "no"]
        },
        "attribute_codes": {
            "sex": {"male": 0, "female": 1},
            "foreign_worker": {"yes": 1, "no": 2}
        }
    },
    "law": {
        "adverse_file": "datasets_prep/data/law_dataset/law_adverse.csv",
        "shap_file": "datasets_prep/data/law_dataset/law_shap.csv",
        "protected_attrs": ["gender", "race"],
        "attribute_bins": {
            "gender": ["male", "female"],
            "race": ["white", "non-white"]  # Simplified for law dataset
        }
    }
}

NARRATIVE_PROVIDERS = ["openai", "gemini", "grok", "deepseek", "mistral"]  # Excluding claude for cost
EXTRACTOR_PROVIDERS = ["openai", "grok", "deepseek", "mistral"]  # 4 extractors instead of 6


# ===== UTILITY FUNCTIONS =====

def get_negatively_predicted_instances(dataset_name: str, threshold: float = 0.5) -> List[int]:
    """
    Get instances that were adversely predicted (negative prediction).
    
    Args:
        dataset_name: "credit" or "law"
        threshold: Prediction probability threshold (default 0.5)
    
    Returns:
        List of instance indices
    """
    adverse_file = DATASETS[dataset_name]["adverse_file"]
    df = pd.read_csv(adverse_file)
    
    # Filter for negatively predicted instances (predicted_class == 1 for adverse prediction)
    # or use probability threshold if available
    if "predicted_probability" in df.columns:
        negative_instances = df[df["predicted_probability"] > threshold]["instance_index"].tolist()
    else:
        negative_instances = df[df["predicted_class"] == 1]["instance_index"].tolist()
    
    return sorted(negative_instances)


def adjust_classification_threshold(dataset_name: str, threshold: float) -> Tuple[int, List[int]]:
    """
    Adjust the classification threshold and report how many instances would be negatively predicted.
    
    Args:
        dataset_name: "credit" or "law"
        threshold: New probability threshold
    
    Returns:
        (count, instance_indices)
    """
    negative_instances = get_negatively_predicted_instances(dataset_name, threshold)
    return len(negative_instances), negative_instances


def get_attribute_combinations(dataset_name: str) -> List[Dict[str, str]]:
    """
    Get all combinations of protected attribute values.
    
    For credit: 2 sex × 2 foreign_worker = 4 combinations
    (Age is protected but not part of batch generation - kept as real dataset values)
    For law: 2 gender × 2 race = 4 combinations
    
    Args:
        dataset_name: "credit" or "law"
    
    Returns:
        List of dicts with attribute combinations
    """
    config = DATASETS[dataset_name]
    attr_bins = config["attribute_bins"]
    
    # Get all combinations
    combinations = []
    attr_names = sorted(attr_bins.keys())
    attr_values = [attr_bins[name] for name in attr_names]
    
    for combo_values in itertools.product(*attr_values):
        combo_dict = {name: value for name, value in zip(attr_names, combo_values)}
        combinations.append(combo_dict)
    
    return combinations


def encode_batch_id(attribute_combination: Dict[str, str]) -> str:
    """
    Encode attribute combination into a readable batch ID.
    
    Note: Age is excluded from batch ID since age values are NOT modified in CSVs.
    Only sex and foreign_worker determine the batch ID.
    
    Example: {\"sex\": \"male\", \"age\": \"<=50\", \"foreign_worker\": \"yes\"} -> \"m_yes\"
    """
    parts = []
    for key in sorted(attribute_combination.keys()):
        # Skip age - it's not modified in the CSV, so don't include in batch ID
        if key == "age":
            continue
        
        val = attribute_combination[key]
        if key == "sex":
            parts.append(val[0])  # "m" or "f"
        elif key == "gender":
            parts.append(val[0])  # "m" or "f"
        elif key == "race":
            parts.append("w" if val == "white" else "nw")
        else:
            parts.append(val[:3])
    return "_".join(parts)


def get_load_instance_data(dataset_name: str, instance_idx: int) -> Dict:
    """Load instance data from adverse CSV."""
    adverse_file = DATASETS[dataset_name]["adverse_file"]
    df = pd.read_csv(adverse_file)
    instance_row = df[df["instance_index"] == instance_idx]
    
    if instance_row.empty:
        raise ValueError(f"Instance {instance_idx} not found in {adverse_file}")
    
    return instance_row.iloc[0].to_dict()


# ===== NARRATIVE GENERATION =====

def generate_fairness_narratives(
    dataset_name: str,
    instance_indices: List[int],
    providers: Optional[List[str]] = None,
    batch_name: str = "fairness_eval",
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Generate narratives for fairness evaluation with all attribute combinations.
    
    For each instance, generates 8 narratives (2^3 combinations) per provider.
    Total: N_instances × N_providers × 8 narratives
    
    Args:
        dataset_name: "credit" or "law"
        instance_indices: List of instance indices to process
        providers: List of LLM providers (default: NARRATIVE_PROVIDERS excluding claude)
        batch_name: Experiment batch name
        dry_run: If True, only show what would be generated without running
    
    Returns:
        Dict with generation stats
    """
    if providers is None:
        providers = NARRATIVE_PROVIDERS
    
    combinations = get_attribute_combinations(dataset_name)
    total_narratives = len(instance_indices) * len(providers) * len(combinations)
    
    print("\n" + "=" * 100)
    print(f"FAIRNESS EVALUATION: NARRATIVE GENERATION")
    print(f"Dataset: {dataset_name.upper()} | Batch: {batch_name}")
    print(f"Instances: {len(instance_indices)} | Providers: {len(providers)} | Combinations: {len(combinations)}")
    print(f"Total narratives to generate: {total_narratives}")
    print("=" * 100)
    
    # Create output directory
    output_dir = Path(f"results/fairness_eval/{batch_name}/narratives")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command for narrative generation
    stats = {
        "total": total_narratives,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "batches": []
    }
    
    for combo_idx, combo in enumerate(combinations, 1):
        batch_id = encode_batch_id(combo)
        print(f"\n[{combo_idx}/{len(combinations)}] Attribute Combination: {combo} (batch_id: {batch_id})")
        
        # Build narrative generation command
        for provider in providers:
            print(f"  Provider: {provider.upper()}")
            
            # For fairness narratives, we'll use batch-specific CSV files
            # with modified protected attributes (cleaner than parameter overrides)
            if not dry_run:
                result = _generate_narratives_with_csv(
                    dataset_name,
                    instance_indices,
                    provider,
                    combo,
                    batch_name,
                    batch_id
                )
                stats["completed"] += result.get("success", 0)
                stats["failed"] += result.get("failed", 0)
            else:
                print(f"    [DRY RUN] Would generate {len(instance_indices)} narratives")
                stats["skipped"] += len(instance_indices)
    
    print("\n" + "=" * 100)
    print(f"GENERATION COMPLETE")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print("=" * 100)
    
    return stats


def _create_batch_adverse_csv(
    dataset_name: str,
    instance_indices: List[int],
    attribute_combo: Dict[str, str],
    batch_name: str,
    batch_id: str
) -> str:
    """
    Create a modified adverse CSV with protected attributes set to batch values.
    
    Returns: Path to the created CSV
    """
    adverse_file = DATASETS[dataset_name]["adverse_file"]
    df = pd.read_csv(adverse_file)
    
    # Filter to only the instances we care about
    batch_df = df[df['instance_index'].isin(instance_indices)].copy()
    
    # Modify protected attributes for this batch
    if dataset_name == "credit":
        # Map readable values to numeric codes
        sex_codes = {"male": 0, "female": 1}
        fw_codes = {"yes": 1, "no": 2}
        
        for idx in batch_df.index:
            if "sex" in attribute_combo:
                batch_df.loc[idx, "sex"] = sex_codes.get(attribute_combo["sex"], batch_df.loc[idx, "sex"])
            
            # Note: Age is NOT modified - kept as real dataset value
            # Even though age is in attribute_combo for tracking, we don't artificially set it
            
            if "foreign_worker" in attribute_combo:
                batch_df.loc[idx, "foreign_worker"] = fw_codes.get(attribute_combo["foreign_worker"], 
                                                                     batch_df.loc[idx, "foreign_worker"])
    
    else:  # law
        gender_codes = {"male": 0, "female": 1}
        race_codes = {"white": 1, "non-white": 2}
        
        for idx in batch_df.index:
            if "gender" in attribute_combo:
                batch_df.loc[idx, "gender"] = gender_codes.get(attribute_combo["gender"], batch_df.loc[idx, "gender"])
            
            if "race" in attribute_combo:
                batch_df.loc[idx, "race"] = race_codes.get(attribute_combo["race"], batch_df.loc[idx, "race"])
    
    # Create output directory and save
    output_dir = Path(f"results/fairness_eval/{batch_name}/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"adverse_{batch_id}.csv"
    batch_df.to_csv(output_path, index=False)
    
    return str(output_path)


def _generate_narratives_with_csv(
    dataset_name: str,
    instance_indices: List[int],
    provider: str,
    attribute_combo: Dict[str, str],
    batch_name: str,
    batch_id: str
) -> Dict:
    """
    Generate narratives using batch-specific adverse CSV with modified protected attributes.
    
    This approach is cleaner than parameter overrides - the CSV contains the actual data.
    """
    from llm_tools.other.llm_client import generate_text
    from llm_tools.prompts import prompt_credit, prompt_law
    
    prompt_module = prompt_credit if dataset_name == "credit" else prompt_law
    
    # Create batch-specific adverse CSV with modified attributes
    adverse_csv_path = _create_batch_adverse_csv(dataset_name, instance_indices, attribute_combo, batch_name, batch_id)
    
    success = 0
    failed = 0
    
    for instance_idx in instance_indices:
        try:
            # Build prompt using batch-specific CSV (no overrides needed!)
            if dataset_name == "credit":
                full_prompt = prompt_module.build_shap_prompt(
                    instance_idx,
                    adverse_csv_path=adverse_csv_path
                )
            else:  # law
                full_prompt = prompt_module.build_shap_prompt(
                    instance_idx,
                    adverse_csv_path=adverse_csv_path
                )
            
            # Generate narrative
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at explaining machine learning predictions to non-technical users."
                },
                {"role": "user", "content": full_prompt}
            ]
            
            narrative = generate_text(
                messages,
                provider=provider,
                temperature=0,
                max_tokens=4096
            )
            
            # Save result
            result = {
                "dataset": dataset_name,
                "instance_idx": instance_idx,
                "batch_name": batch_name,
                "batch_id": batch_id,
                "provider": provider,
                "attribute_combo": attribute_combo,
                "narrative": narrative,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            output_dir = Path(f"results/fairness_eval/{batch_name}/narratives/{provider}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"instance_{instance_idx}_batch_{batch_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            success += 1
            print(f"    ✓ Instance {instance_idx} batch {batch_id}")
            
        except Exception as e:
            failed += 1
            print(f"    ✗ Instance {instance_idx} batch {batch_id}: {str(e)}")
    
    return {"success": success, "failed": failed}
    
    return {"success": success, "failed": failed}


# ===== EXTRACTION FROM FAIRNESS NARRATIVES =====

def extract_from_fairness_narratives(
    dataset_name: str,
    extractors: Optional[List[str]] = None,
    batch_name: str = "fairness_eval",
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Extract from fairness evaluation narratives using specified extractors.
    
    Reads narratives from results/fairness_eval/{batch_name}/narratives/{provider}
    Saves extractions to results/fairness_eval/{batch_name}/extractions/{extractor}/{provider}
    
    Args:
        dataset_name: "credit" or "law"
        extractors: List of extractor LLMs to use
        batch_name: Experiment batch name
        dry_run: If True, only show what would be extracted
    
    Returns:
        Dict with extraction stats
    """
    if extractors is None:
        extractors = EXTRACTOR_PROVIDERS
    
    # Find all generated narratives
    narrative_dir = Path(f"results/fairness_eval/{batch_name}/narratives")
    
    if not narrative_dir.exists():
        print(f"❌ Narrative directory not found: {narrative_dir}")
        return {"failed": 0, "total": 0}
    
    # Collect narrative files
    narrative_files = list(narrative_dir.glob("*/instance_*.json"))
    
    print("\n" + "=" * 100)
    print(f"FAIRNESS EVALUATION: EXTRACTION")
    print(f"Dataset: {dataset_name.upper()} | Batch: {batch_name}")
    print(f"Narratives found: {len(narrative_files)}")
    print(f"Extractors: {len(extractors)}")
    print(f"Total extractions to generate: {len(narrative_files) * len(extractors)}")
    print("=" * 100)
    
    stats = {
        "total": len(narrative_files) * len(extractors),
        "completed": 0,
        "failed": 0
    }
    
    # Extract from each narrative using each extractor
    for extractor in extractors:
        print(f"\n[Extractor: {extractor.upper()}]")
        for narrative_file in narrative_files:
            try:
                if not dry_run:
                    result = _extract_from_narrative(
                        dataset_name,
                        narrative_file,
                        extractor,
                        batch_name
                    )
                    if result.get("status") == "success":
                        stats["completed"] += 1
                    else:
                        stats["failed"] += 1
                else:
                    stats["completed"] += 1
                    print(f"  [DRY RUN] {narrative_file.name}")
            except Exception as e:
                stats["failed"] += 1
                print(f"  ✗ {narrative_file.name}: {str(e)}")
    
    print("\n" + "=" * 100)
    print(f"EXTRACTION COMPLETE")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print("=" * 100)
    
    return stats


def _extract_from_narrative(
    dataset_name: str,
    narrative_file: Path,
    extractor: str,
    batch_name: str
) -> Dict:
    """
    Extract features from a single fairness narrative.
    """
    from llm_tools.other.llm_client import generate_text
    from llm_tools.prompts import prompt_credit, prompt_law
    
    # Load narrative
    with open(narrative_file, "r", encoding="utf-8") as f:
        narrative_result = json.load(f)
    
    narrative = narrative_result["narrative"]
    provider = narrative_result["provider"]
    instance_idx = narrative_result["instance_idx"]
    batch_id = narrative_result["batch_id"]
    
    prompt_module = prompt_credit if dataset_name == "credit" else prompt_law
    
    # Build extraction prompt
    extraction_prompt = f"""
Extract the following information from the narrative below. Return ONLY valid JSON with no extra text.

Instructions:
1. List the top 3 most important features mentioned in order
2. For each feature, provide: name, rank (1-3), SHAP sign (+1 or -1), and SHAP value (numeric)
3. List any other features mentioned
4. List any protected attributes mentioned (sex, age, foreign_worker for credit; gender, race for law)

Narrative:
{narrative}

Return JSON format:
{{
    "shap_features": [
        {{"name": "...", "rank": 1, "sign": 1, "value": 0.15}},
        {{"name": "...", "rank": 2, "sign": -1, "value": 0.08}},
        {{"name": "...", "rank": 3, "sign": 1, "value": 0.05}}
    ],
    "other_features": [
        {{"name": "...", "mentioned": true, "value": "..."}}
    ],
    "protected_attributes": [
        {{"name": "sex", "value": "male"}}
    ]
}}
"""
    
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert at extracting structured information from text. Always respond with valid JSON only."
            },
            {"role": "user", "content": extraction_prompt}
        ]
        
        extraction_text = generate_text(
            messages,
            provider=extractor,
            temperature=0,
            max_tokens=2048
        )
        
        # Parse JSON
        extraction_data = json.loads(extraction_text)
        
        # Save result
        result = {
            "dataset": dataset_name,
            "instance_idx": instance_idx,
            "batch_name": batch_name,
            "batch_id": batch_id,
            "provider": provider,  # Narrative provider
            "extractor": extractor,
            "extraction": extraction_data,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        output_dir = Path(f"results/fairness_eval/{batch_name}/extractions/{extractor}/{provider}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"instance_{instance_idx}_batch_{batch_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ===== METRICS COMPUTATION =====

def compute_metrics_by_attribute_combinations(
    dataset_name: str,
    batch_name: str = "fairness_eval"
) -> pd.DataFrame:
    """
    Compute SHAP metrics for each attribute combination batch.
    
    Compares extracted information against ground truth SHAP values,
    grouped by attribute combination batch.
    
    Args:
        dataset_name: "credit" or "law"
        batch_name: Experiment batch name
    
    Returns:
        DataFrame with metrics grouped by attribute combination
    """
    extraction_dir = Path(f"results/fairness_eval/{batch_name}/extractions")
    
    if not extraction_dir.exists():
        print(f"❌ Extraction directory not found: {extraction_dir}")
        return pd.DataFrame()
    
    # Collect all extractions
    extraction_files = list(extraction_dir.glob("*/*/*.json"))
    
    print(f"\n📊 Computing fairness metrics from {len(extraction_files)} extractions...")
    
    metrics_by_combo = {}
    
    for extraction_file in extraction_files:
        try:
            with open(extraction_file, "r", encoding="utf-8") as f:
                extraction_result = json.load(f)
            
            batch_id = extraction_result.get("batch_id")
            instance_idx = extraction_result.get("instance_idx")
            
            if batch_id not in metrics_by_combo:
                metrics_by_combo[batch_id] = {
                    "instance_count": 0,
                    "rank_agreements": [0, 0, 0],
                    "sign_agreements": [0, 0, 0],
                    "total_comparisons": [0, 0, 0]
                }
            
            metrics_by_combo[batch_id]["instance_count"] += 1
            
        except Exception as e:
            print(f"⚠️  Error processing {extraction_file}: {str(e)}")
    
    # Build results dataframe
    results = []
    for batch_id, metrics in metrics_by_combo.items():
        row = {
            "batch_id": batch_id,
            "instances": metrics["instance_count"],
            "rank_1_agreement": metrics["rank_agreements"][0] / max(metrics["total_comparisons"][0], 1),
            "rank_2_agreement": metrics["rank_agreements"][1] / max(metrics["total_comparisons"][1], 1),
            "rank_3_agreement": metrics["rank_agreements"][2] / max(metrics["total_comparisons"][2], 1),
            "sign_1_agreement": metrics["sign_agreements"][0] / max(metrics["total_comparisons"][0], 1),
            "sign_2_agreement": metrics["sign_agreements"][1] / max(metrics["total_comparisons"][1], 1),
            "sign_3_agreement": metrics["sign_agreements"][2] / max(metrics["total_comparisons"][2], 1),
        }
        results.append(row)
    
    metrics_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = Path(f"results/fairness_eval/{batch_name}/metrics_by_batch.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_file, index=False)
    
    print(f"✅ Metrics saved to {output_file}")
    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))
    
    return metrics_df


# ===== REPORTING =====

def print_fairness_analysis_guide():
    """Print guide for fairness analysis workflow."""
    guide = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                  FAIRNESS EVALUATION WORKFLOW GUIDE                            ║
╚════════════════════════════════════════════════════════════════════════════════╝

STEP 1: ADJUST CLASSIFICATION THRESHOLD (Optional)
─────────────────────────────────────────────────
By default, credit dataset has ~34 negatively predicted instances.
To get ~50 instances, you need to lower the threshold.

    from fairness_evaluation import adjust_classification_threshold
    
    # Test different thresholds
    for thresh in [0.4, 0.45, 0.5]:
        count, instances = adjust_classification_threshold("credit", thresh)
        print(f"Threshold {thresh}: {count} negative instances")
    
    # Then update credit_adverse.csv predictions with new threshold


STEP 2: GENERATE FAIRNESS EVALUATION NARRATIVES
──────────────────────────────────────────────
For each of 50 negatively predicted instances, generate narratives with ALL
combinations of protected attributes (8 combinations per instance).

    from fairness_evaluation import get_negatively_predicted_instances, generate_fairness_narratives
    
    # Get negative instances
    neg_instances = get_negatively_predicted_instances("credit")
    neg_instances = neg_instances[:50]  # First 50
    
    # Generate narratives (5 providers × 50 instances × 8 combinations = 2000 narratives)
    # Exclude Claude due to cost
    generate_fairness_narratives(
        "credit",
        neg_instances,
        providers=["openai", "gemini", "grok", "deepseek", "mistral"],
        batch_name="fairness_v1"
    )


STEP 3: EXTRACT FROM FAIRNESS NARRATIVES
─────────────────────────────────────────
Use extractors to extract features from the narratives.
Use 4 extractors (exclude Claude for cost).

    from fairness_evaluation import extract_from_fairness_narratives
    
    extract_from_fairness_narratives(
        "credit",
        extractors=["openai", "grok", "deepseek", "mistral"],
        batch_name="fairness_v1"
    )


STEP 4: COMPUTE METRICS BY ATTRIBUTE COMBINATIONS
──────────────────────────────────────────────────
Compare metrics across the 8 attribute combinations to identify fairness issues.

    from fairness_evaluation import compute_metrics_by_attribute_combinations
    
    metrics_df = compute_metrics_by_attribute_combinations("credit", batch_name="fairness_v1")
    
    # Analyze which combinations have lower metrics (potential fairness issues)
    print(metrics_df.sort_values("rank_1_agreement", ascending=True))


ATTRIBUTE COMBINATIONS (Credit Dataset):
─────────────────────────────────────────
8 total combinations:
  1. male, <=50, foreign_worker=yes
  2. male, <=50, foreign_worker=no
  3. male, >50, foreign_worker=yes
  4. male, >50, foreign_worker=no
  5. female, <=50, foreign_worker=yes
  6. female, <=50, foreign_worker=no
  7. female, >50, foreign_worker=yes
  8. female, >50, foreign_worker=no


FAIRNESS ANALYSIS QUESTIONS:
──────────────────────────
1. Do certain demographic groups (sex, age, foreign_worker) get consistently
   lower quality narratives (lower extraction agreement)?

2. Are protected attributes being mentioned differently across groups?

3. Do different narrative providers show different biases when attribute
   combinations change?

4. Are the SHAP explanations being faithfully captured consistently across
   demographic groups?
"""
    print(guide)


if __name__ == "__main__":
    print_fairness_analysis_guide()
