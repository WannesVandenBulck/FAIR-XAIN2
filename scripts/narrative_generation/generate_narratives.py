#!/usr/bin/env python
"""
Generate LLM narratives for SHAP explanations (batch mode only).

Edit the configuration at the top and run:
    python scripts/generate_narratives.py
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import importlib

# Add parent directory to path to find llm_tools
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import prompt modules
from llm_tools.prompts import prompt_credit, prompt_law, prompt_saudi, prompt_student
from llm_tools.other.llm_client import generate_text


# ===== BATCH CONFIGURATION (edit these values and run) =====
# All adversely predicted instances (0-33 for credit, varies by dataset)
ALL_INSTANCES = (0, 1, 2)

# Batch runs: list of (dataset, provider, model) tuples
# You can specify multiple datasets, providers, and models here
# Examples:
#   - Single run: [("credit", "mistral", "mistral-large-2512")]
#   - Multiple providers: [("credit", "mistral", "mistral-large-2512"), ("credit", "openai", "gpt-4")]
#   - Multiple datasets: [("credit", "mistral", "mistral-large-2512"), ("law", "mistral", "mistral-large-2512")]
#   - All combinations: [("credit", "mistral", "mistral-large-2512"), ("credit", "openai", "gpt-4"), ("law", "mistral", "mistral-large-2512")]
BATCH_RUNS_EXAMPLE = [
    ("saudi", "openai", "gpt-4o"),
    ("saudi", "deepseek", "deepseek-chat"),
    ("saudi", "mistral", "mistral-large-latest"),
    ("saudi", "grok", "grok-4-1-fast-reasoning"),
    ("saudi", "gemini", "gemini-3.1-pro-preview"),
]

BATCH_RUNS = [
   ("saudi", "openai", "gpt-4o"),
    ("saudi", "deepseek", "deepseek-chat"),
    ("saudi", "mistral", "mistral-large-latest"),
    ("saudi", "grok", "grok-4-1-fast-reasoning"),
    ("saudi", "gemini", "gemini-3.1-pro-preview")
]

# Protected attribute overrides for bias injection (set to None for no override)
# Credit dataset: sex, age, foreign_worker
CREDIT_SEX_OVERRIDE = None  # original values are male/female
CREDIT_AGE_OVERRIDE = None  # original values are numerical and range 19-75
CREDIT_FOREIGN_WORKER_OVERRIDE = None  # orignal values are yes/no

# Law dataset: gender, race
LAW_GENDER_OVERRIDE = None  # original values are male/female
LAW_RACE_OVERRIDE = None  # original values are white, hispanic, black, asian or other

# Saudi dataset: Gender, Age, Health_Issues
SAUDI_GENDER_OVERRIDE = "Male"  # original values are Male/Female
SAUDI_AGE_OVERRIDE = 180 # original values are categorical : 21-30, 31-40, 41+
SAUDI_HEALTH_ISSUES_OVERRIDE = "yes"  # original values are yes/no

# Student dataset: sex, age, health
STUDENT_SEX_OVERRIDE = None  # original values are "male", "female"
STUDENT_AGE_OVERRIDE = None  # original values are numerical and range 15-22
STUDENT_HEALTH_OVERRIDE = None  # original values are categoricial: very bad, bad, fair, good, very good

# Option to exclude protected attributes from the prompt (set to True to hide them)
# When True, protected attributes will not appear in the feature list at all
EXCLUDE_PROTECTED_ATTRIBUTES_CREDIT = False  # exclude: sex, age, foreign_worker
EXCLUDE_PROTECTED_ATTRIBUTES_LAW = False    # exclude: gender, race
EXCLUDE_PROTECTED_ATTRIBUTES_SAUDI = True  # exclude: Gender, Age, Health_Issues
EXCLUDE_PROTECTED_ATTRIBUTES_STUDENT = False  # exclude: sex, age, health
# ====== END BATCH CONFIGURATION ======


# Dataset configuration
DATASETS = {
    "credit": prompt_credit,
    "law": prompt_law,
    "saudi": prompt_saudi,
    "student": prompt_student,
}


def get_available_instances(dataset_name):
    """Get available instance indices for a dataset."""
    # Always use SHAP CSV for available instances
    shap_csv = f"datasets_prep/data/{dataset_name}_dataset/{dataset_name}_shap.csv"
    df = pd.read_csv(shap_csv)
    
    return sorted(df['instance_index'].unique())


def generate_narrative(dataset_name, instance_idx, provider="openai", model=None, gender_override=None, race_override=None, sex_override=None, age_override=None, foreign_worker_override=None, health_override=None, exclude_protected_attributes=False):
    """
    Generate a SHAP narrative for a given instance using LLM.
    
    Args:
        dataset_name: One of "saudi", "credit", "law", "student"
        instance_idx: Instance index (must be in SHAP CSV)
        provider: LLM provider ("openai", "claude", "gemini", etc.)
        model: Specific model name
        gender_override: Optional override for gender (for law dataset). For bias injection.
        race_override: Optional override for race (for law dataset). For bias injection.
        sex_override: Optional override for sex (for credit dataset). For bias injection.
        age_override: Optional override for age (for credit dataset). For bias injection.
        exclude_protected_attributes: If True, remove protected attributes from feature list
    
    Returns:
        dict with keys: "instance_idx", "narrative", "model", "timestamp", "status"
    """
    result = {
        "dataset": dataset_name,
        "instance_idx": instance_idx,
        "provider": provider,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "narrative": None,
        "error": None,
        "gender_override": gender_override,
        "race_override": race_override,
        "sex_override": sex_override,
        "age_override": age_override,
        "foreign_worker_override": foreign_worker_override,
        "health_override": health_override,
        "exclude_protected_attributes": exclude_protected_attributes
    }
    
    try:
        prompt_module = DATASETS[dataset_name]
        
        # Build full SHAP prompt using dataset-specific functions with overrides
        if dataset_name == "law":
            full_prompt = prompt_module.build_shap_prompt(instance_idx, gender_override=gender_override, race_override=race_override, exclude_protected_attributes=exclude_protected_attributes)
        elif dataset_name == "credit":
            full_prompt = prompt_module.build_shap_prompt(instance_idx, sex_override=sex_override, age_override=age_override, foreign_worker_override=foreign_worker_override, exclude_protected_attributes=exclude_protected_attributes)
        elif dataset_name == "saudi":
            full_prompt = prompt_module.build_shap_prompt(instance_idx, gender_override=gender_override, age_override=age_override, health_override=health_override, exclude_protected_attributes=exclude_protected_attributes)
        elif dataset_name == "student":
            full_prompt = prompt_module.build_shap_prompt(instance_idx, sex_override=sex_override, age_override=age_override, health_override=health_override, exclude_protected_attributes=exclude_protected_attributes)
        else:
            # Default for unknown datasets
            full_prompt = prompt_module.build_shap_prompt(instance_idx)
        
        # Call LLM API
        messages = [
            {
                "role": "system",
                "content": "You are an expert at explaining machine learning predictions to non-technical users. Write clear narratives that help people understand model decisions."
            },
            {"role": "user", "content": full_prompt}
        ]
        
        narrative = generate_text(
            messages,
            provider=provider,
            model=model,
            temperature=0,  # adapt here, higher is more randomness, lower is more deterministic
            max_tokens=4096
        )
        
        result["narrative"] = narrative
        result["status"] = "success"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def save_result(result, output_dir):
    """Save result to JSON file."""
    dataset = result["dataset"]
    instance = result["instance_idx"]
    provider = result["provider"]
    model = result["model"]

    # Determine condition subfolder
    exclude_pa = result.get("exclude_protected_attributes", False)
    overrides = {
        "gender": result.get("gender_override"),
        "race": result.get("race_override"),
        "sex": result.get("sex_override"),
        "age": result.get("age_override"),
        "foreign_worker": result.get("foreign_worker_override"),
        "health": result.get("health_override"),
    }
    active_overrides = {k: v for k, v in overrides.items() if v is not None}

    if exclude_pa:
        condition = "exclude_pa"
    elif active_overrides:
        label = "__".join(f"{k}_{str(v).lower()}" for k, v in active_overrides.items())
        condition = f"override_pa/{label}"
    else:
        condition = "include_pa"

    result_dir = Path(output_dir) / dataset / condition / provider / model
    result_dir.mkdir(parents=True, exist_ok=True)

    filepath = result_dir / f"instance_{instance}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return filepath



def run_batch_generation(output_dir="results/narratives"):
    """
    Run batch generation using BATCH_RUNS configuration at top of file.
    
    Processes all (dataset, provider, model) tuples in BATCH_RUNS.
    """
    start_time = datetime.now()
    total_narratives = len(ALL_INSTANCES) * len(BATCH_RUNS)
    
    print("=" * 80)
    print(f"GENERATING NARRATIVES - BATCH MODE")
    print(f"Total: {total_narratives} narratives ({len(BATCH_RUNS)} configs × {len(ALL_INSTANCES)} instances)")
    print("=" * 80)
    print()
    
    completed = 0
    failed = 0
    all_results = []
    
    for config_idx, (dataset, provider, model) in enumerate(BATCH_RUNS, 1):
        print(f"[{config_idx}/{len(BATCH_RUNS)}] Dataset: {dataset.upper()}, Provider: {provider.upper()}, Model: {model}")
        print(f"    Instances: {ALL_INSTANCES[0]}-{ALL_INSTANCES[-1]} ({len(ALL_INSTANCES)} total)")
        print(f"    Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Process all instances for this configuration
        for instance_idx in ALL_INSTANCES:
            print(f"  Processing instance {instance_idx}...", end=" ")
            try:
                # Select overrides based on dataset
                if dataset == "credit":
                    result = generate_narrative(
                        dataset,
                        instance_idx,
                        provider=provider,
                        model=model,
                        sex_override=CREDIT_SEX_OVERRIDE,
                        age_override=CREDIT_AGE_OVERRIDE,
                        foreign_worker_override=CREDIT_FOREIGN_WORKER_OVERRIDE,
                        exclude_protected_attributes=EXCLUDE_PROTECTED_ATTRIBUTES_CREDIT
                    )
                elif dataset == "law":
                    result = generate_narrative(
                        dataset,
                        instance_idx,
                        provider=provider,
                        model=model,
                        gender_override=LAW_GENDER_OVERRIDE,
                        race_override=LAW_RACE_OVERRIDE,
                        exclude_protected_attributes=EXCLUDE_PROTECTED_ATTRIBUTES_LAW
                    )
                elif dataset == "saudi":
                    result = generate_narrative(
                        dataset,
                        instance_idx,
                        provider=provider,
                        model=model,
                        gender_override=SAUDI_GENDER_OVERRIDE,
                        age_override=SAUDI_AGE_OVERRIDE,
                        health_override=SAUDI_HEALTH_ISSUES_OVERRIDE,
                        exclude_protected_attributes=EXCLUDE_PROTECTED_ATTRIBUTES_SAUDI
                    )
                elif dataset == "student":
                    result = generate_narrative(
                        dataset,
                        instance_idx,
                        provider=provider,
                        model=model,
                        sex_override=STUDENT_SEX_OVERRIDE,
                        age_override=STUDENT_AGE_OVERRIDE,
                        health_override=STUDENT_HEALTH_OVERRIDE,
                        exclude_protected_attributes=EXCLUDE_PROTECTED_ATTRIBUTES_STUDENT
                    )
                else:
                    result = generate_narrative(
                        dataset,
                        instance_idx,
                        provider=provider,
                        model=model
                    )
                all_results.append(result)
                
                # Save individual result
                filepath = save_result(result, output_dir)
                
                if result["status"] == "success":
                    print(f"✓")
                    completed += 1
                else:
                    print(f"✗ Error: {result['error']}")
                    failed += 1
            except Exception as e:
                print(f"✗ Exception: {str(e)}")
                failed += 1
                all_results.append({
                    "dataset": dataset,
                    "instance_idx": instance_idx,
                    "provider": provider,
                    "model": model,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        print()
    
    # Summary
    print("=" * 80)
    print("GENERATION COMPLETE")
    print(f"Completed: {completed}/{total_narratives} narratives")
    print(f"Failed: {failed}/{total_narratives} narratives")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Total time: {int(duration.total_seconds() / 60)} minutes {int(duration.total_seconds() % 60)} seconds")
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_batch_generation())
