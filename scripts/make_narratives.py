"""
Generate LLM narratives for SHAP and Counterfactual explanations.

Usage:
    python scripts/make_narratives.py --dataset saudi --prompt-type shap --instances 62 63 71
    python scripts/make_narratives.py --dataset credit --prompt-type cf --all-instances
    python scripts/make_narratives.py --dataset law --prompt-type shap --model claude-3-opus
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import importlib
import pickle

try:
    dill = importlib.import_module("dill")
except ModuleNotFoundError:
    dill = pickle

# Add parent directory to path to find llm_tools
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import prompt modules
from llm_tools.prompts import prompt_credit, prompt_law
from llm_tools.other.llm_client import generate_text


# Dataset configuration
DATASETS = {
    "credit": prompt_credit,
    "law": prompt_law,
}

# LLM provider configuration
LLM_PROVIDERS = ["openai", "claude", "gemini", "grok", "deepseek", "mistral"]
LLM_MODELS = {
    "openai": ["gpt-4o"],
    "claude": ["claude-sonnet-4-6"],
    "gemini": ["gemini-3-flash-preview"],
    "grok": ["grok-4-1-fast-non-reasoning"],
    "deepseek": ["deepseek-chat"],
    "mistral": ["mistral-large-2512"],
}


def get_available_instances(dataset_name, prompt_type):
    """Get available instance indices for a dataset and prompt type."""
    prompt_module = DATASETS[dataset_name]
    
    if prompt_type == "shap": 
        # Both shap and narrative use the same SHAP CSV
        shap_csv = f"datasets_prep/data/{dataset_name}_dataset/{dataset_name}_shap.csv"
        df = pd.read_csv(shap_csv)
    else:  # counterfactual
        cf_csv = f"datasets_prep/data/{dataset_name}_dataset/{dataset_name}_counterfactual.csv"
        df = pd.read_csv(cf_csv)
    
    return sorted(df['instance_index'].unique())


def generate_narrative(dataset_name, instance_idx, prompt_type, provider="openai", model=None, gender_override=None, race_override=None, sex_override=None, age_override=None):
    """
    Generate a narrative for a given instance using LLM.
    
    Args:
        dataset_name: One of "saudi", "credit", "law", "student"
        instance_idx: Instance index (must be in SHAP or CF CSV)
        prompt_type: "shap" or "cf"
        provider: LLM provider ("openai", "anthropic", "ollama")
        model: Specific model name
        gender_override: Optional override for gender (for law dataset). For bias injection.
        race_override: Optional override for race (for law dataset). For bias injection.
        sex_override: Optional override for sex (for credit dataset). For bias injection.
        age_override: Optional override for age (for credit dataset). For bias injection.
        sex_override: Optional override for sex (for credit dataset). For bias injection.
        age_override: Optional override for age (for credit dataset). For bias injection.
    
    Returns:
        dict with keys: "instance_idx", "prompt_type", "narrative", "model", "timestamp", "status"
    """
    result = {
        "dataset": dataset_name,
        "instance_idx": instance_idx,
        "prompt_type": prompt_type,
        "provider": provider,
        "model": model or (
            "gpt-4o" if provider == "openai" 
            else "claude-sonnet-4-6" if provider == "claude"
            else "gemini-3-flash-preview" if provider == "gemini"
            else "grok-4-1-fast-non-reasoning" if provider == "grok"
            else "deepseek-chat" if provider == "deepseek"
            else "mistral-large-latest" if provider == "mistral"
            else "gpt-4o"
        ),
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "narrative": None,
        "error": None,
        "gender_override": gender_override,
        "race_override": race_override,
        "sex_override": sex_override,
        "age_override": age_override
    }
    
    try:
        prompt_module = DATASETS[dataset_name]
        
        # Build full prompt using dataset-specific functions
        if prompt_type == "shap":
            # Prepare dataset-specific overrides
            if dataset_name == "law":
                full_prompt = prompt_module.build_shap_prompt(instance_idx, gender_override=gender_override, race_override=race_override)
            elif dataset_name == "credit":
                full_prompt = prompt_module.build_shap_prompt(instance_idx, sex_override=sex_override, age_override=age_override)
        elif prompt_type == "narrative":
            # Use narrative variant if available, otherwise fall back to standard SHAP
            if hasattr(prompt_module, 'build_shap_prompt_narrative'):
                full_prompt = prompt_module.build_shap_prompt_narrative(instance_idx)
            else:
                print(f"Warning: Narrative prompt not available for {dataset_name}, using standard SHAP")
                if dataset_name == "law":
                    full_prompt = prompt_module.build_shap_prompt(instance_idx, gender_override=gender_override, race_override=race_override)
                elif dataset_name == "credit":
                    full_prompt = prompt_module.build_shap_prompt(instance_idx, sex_override=sex_override, age_override=age_override)
                else:
                    full_prompt = prompt_module.build_shap_prompt(instance_idx, gender_override=gender_override, race_override=race_override)
        elif prompt_type == "cf":
            # Prepare dataset-specific overrides
            if dataset_name == "law":
                full_prompt = prompt_module.build_cf_prompt(instance_idx, gender_override=gender_override, race_override=race_override)
            elif dataset_name == "credit":
                full_prompt = prompt_module.build_cf_prompt(instance_idx, sex_override=sex_override, age_override=age_override)
            else:
                # Default behavior for other datasets
                full_prompt = prompt_module.build_cf_prompt(instance_idx, gender_override=gender_override, race_override=race_override)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
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
            model=result["model"],
            temperature=0, #adapt here, higher is more randomness, lower is more deterministic
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
    prompt_type = result["prompt_type"]
    provider = result["provider"]
    model = result["model"]
    
    # Create directory structure: dataset/narratives/prompt_type/provider/model
    result_dir = Path(output_dir) / dataset / "narratives" / prompt_type / provider / model
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    filepath = result_dir / f"instance_{instance}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return filepath


def save_experiment_pickle(results, args):
    """Save generated narratives in one aggregated experiment.pkl (SHAPnarrative-metrics style)."""
    model_name = args.model or (
        "gpt-4o" if args.provider == "openai"
        else "claude-sonnet-4-6" if args.provider == "claude"
        else "gemini-3-flash-preview" if args.provider == "gemini"
        else "grok-4-1-fast-non-reasoning" if args.provider == "grok"
        else "deepseek-chat" if args.provider == "deepseek"
        else "mistral-large-latest" if args.provider == "mistral"
        else "gpt-4o"
    )

    # Match reference style: one binary file with all generated narratives for a run.
    experiment_dir = Path(args.output_dir) / "experiments" / args.dataset / args.prompt_type / args.provider / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experiment_payload = {
        "dataset": args.dataset,
        "prompt_type": args.prompt_type,
        "provider": args.provider,
        "model": model_name,
        "run_timestamp": datetime.now().isoformat(),
        "num_instances": len(results),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
        "instances": sorted([r.get("instance_idx") for r in results if "instance_idx" in r]),
        "results": results,
    }

    experiment_path = experiment_dir / "experiment.pkl"
    with open(experiment_path, "wb") as f:
        dill.dump(experiment_payload, f)

    # Optional temp checkpoint path, similar to reference repo usage.
    temp_dir = Path(args.output_dir) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "latest_experiment.pkl"
    with open(temp_path, "wb") as f:
        dill.dump(experiment_payload, f)

    return experiment_path, temp_path


def main():
    parser = argparse.ArgumentParser(description="Generate LLM narratives for SHAP/CF explanations")
    
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="Dataset to process")
    parser.add_argument("--prompt-type", required=True, choices=["shap", "narrative", "cf"],
                        help="Prompt type (shap, narrative, or cf)")
    parser.add_argument("--instances", nargs="+", type=int, default=None,
                        help="Specific instance indices to process")
    parser.add_argument("--all-instances", action="store_true",
                        help="Process all available instances")
    parser.add_argument("--provider", choices=LLM_PROVIDERS, default="openai",
                        help="LLM provider")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name")
    parser.add_argument("--output-dir", default="results/narratives",
                        help="Output directory for results")
    parser.add_argument("--gender-override", type=str, default=None,
                        help="Override gender for all instances (e.g., 'male', 'female'). For bias injection experiment.")
    parser.add_argument("--race-override", type=str, default=None,
                        help="Override race for all instances (e.g., 'white', 'black', 'hispanic'). For bias injection experiment.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without calling LLM")
    
    args = parser.parse_args()
    
    # Determine instances to process
    if args.all_instances:
        instances = get_available_instances(args.dataset, args.prompt_type)
    elif args.instances:
        instances = args.instances
    else:
        parser.error("Must specify --instances or --all-instances")
    
    print(f"Dataset: {args.dataset}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or (
        'gpt-4o' if args.provider == 'openai' 
        else 'claude-sonnet-4-6' if args.provider == 'claude'
        else 'gemini-3-flash-preview' if args.provider == 'gemini'
        else 'grok-4-1-fast-non-reasoning' if args.provider == 'grok'
        else 'deepseek-chat' if args.provider == 'deepseek'
        else 'mistral-large-latest' if args.provider == 'mistral'
        else 'gpt-4o'
    )}")
    print(f"Instances to process: {len(instances)} instances")
    print(f"Output directory: {args.output_dir}")
    if args.gender_override or args.race_override:
        print(f"Bias injection overrides:")
        if args.gender_override:
            print(f"  - Gender: {args.gender_override}")
        if args.race_override:
            print(f"  - Race: {args.race_override}")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would process instances: {instances[:10]}{'...' if len(instances) > 10 else ''}")
        return
    
    # Process instances
    results = []
    for i, instance_idx in enumerate(instances, 1):
        print(f"\n[{i}/{len(instances)}] Processing instance {instance_idx}...", end=" ")
        
        try:
            result = generate_narrative(
                args.dataset,
                instance_idx,
                args.prompt_type,
                provider=args.provider,
                model=args.model,
                gender_override=args.gender_override,
                race_override=args.race_override
            )
            results.append(result)
            
            # Save individual result
            filepath = save_result(result, args.output_dir)
            
            if result["status"] == "success":
                print(f"✓ Saved to {filepath}")
            else:
                print(f"✗ Error: {result['error']}")
        
        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            results.append({
                "dataset": args.dataset,
                "instance_idx": instance_idx,
                "prompt_type": args.prompt_type,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    
    print(f"\n{'='*60}")
    print(f"Summary: {successful} successful, {failed} failed out of {len(results)} total")
    print(f"Results saved to: {args.output_dir}")

    experiment_path, temp_path = save_experiment_pickle(results, args)
    print(f"Aggregated experiment saved to: {experiment_path}")
    print(f"Latest temp checkpoint saved to: {temp_path}")


if __name__ == "__main__":
    main()
