"""
Extraction script: Extract features and values from narratives using LLM.

Edit the CONFIGURATION section below to set:
- DATASETS_TO_PROCESS: which dataset(s) to process (string or list)
- NARRATIVE_PROVIDERS_TO_USE: which narrative providers to extract
- EXTRACTOR_PROVIDERS_TO_USE: which LLM(s) to use for extraction
- INSTANCE_INDICES: which instances to process (list or "all")

Then just run:
    python scripts/extraction.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import signal
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Enable unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)


def timeout(seconds=90):
    """Decorator to timeout a function after N seconds (works on Windows and Unix)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    return False, None, f"Timeout after {seconds} seconds"
        return wrapper
    return decorator

# Add parent path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.extractions.generate_extractor_prompt import generate_extractor_prompt
from llm_tools.other.llm_client import generate_text

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

# Datasets to process (string for single, list for multiple):
#   Examples:
#   DATASETS_TO_PROCESS = "credit"                                    # Single dataset
#   DATASETS_TO_PROCESS = ["credit", "law", "saudi", "student"]      # All datasets
DATASETS_TO_PROCESS = ["law"]  # "credit", "law", "saudi", "student", or list

NARRATIVE_PROVIDERS_TO_USE = ["deepseek"]  # All narrative providers: "gemini", "grok", "deepseek", "mistral", "openai", "claude"
EXTRACTOR_PROVIDERS_TO_USE = ["grok"]  # LLM(s) to use for extraction: "openai", "claude", "gemini", "grok", "deepseek", "mistral"
INSTANCE_INDICES = [31]  # Specific instances, or use "all" to process all instances for each dataset

# Narrative condition to extract from: "include_pa", "exclude_pa", or "override_pa/<label>" (e.g. "override_pa/gender_female__race_black")
# Set to None to search across all conditions.
#NARRATIVE_CONDITION = "override_pa/gender_female__race_black"
NARRATIVE_CONDITION = "exclude_pa"

# ============================================================================

# LLM provider configuration
LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-3-flash-preview",
    "grok": "grok-4.20-0309-non-reasoning",
    "deepseek": "deepseek-v4-flash",
    "mistral": "mistral-large-latest",
}

DATASETS = {
    "credit": {"num_instances": 97},
    "law": {"num_instances": 308},
    "saudi": {"num_instances": 106},
    "student": {"num_instances": 73}
}


@timeout(seconds=90)
def extract_single(dataset_name="credit", instance_idx=0, narrative_provider="gemini",
                   extractor_provider="grok", condition=None):
    """
    Extract features from a single SHAP narrative.
    
    Args:
        dataset_name: "credit", "law", "saudi", or "student"
        instance_idx: Instance number to extract
        narrative_provider: Provider whose narrative to use
        extractor_provider: Provider to use for extraction
    
    Returns:
        (success: bool, extraction: dict or None, error: str or None)
    """
    
    try:
        # Generate the extractor prompt
        prompt = generate_extractor_prompt(dataset_name, instance_idx, narrative_provider, "shap", condition=condition)
        
        if not prompt or prompt.startswith("Error:"):
            return False, None, f"Failed to generate prompt: {prompt}"
        
        # Prepare messages for LLM
        messages = [
            {
                "role": "system",
                "content": "You are an expert data analyst. Your task is to extract information from narratives and fill in a JSON template. Return ONLY the completed JSON in a code block - no other text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Call LLM
        model = LLM_MODELS[extractor_provider]
        response = generate_text(
            messages=messages,
            provider=extractor_provider,
            model=model,
            temperature=0,
            max_tokens=10000
        )
        
        # Save raw response
        raw_response_file = f"results/extractions/{dataset_name}/raw/shap/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.txt"
        os.makedirs(os.path.dirname(raw_response_file), exist_ok=True)
        with open(raw_response_file, "w", encoding="utf-8") as f:
            f.write(response)
        
        # Extract JSON from response
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            if json_end == -1:
                json_end = len(response)
            json_content = response[json_start:json_end].strip()
            
            # Parse and validate JSON
            extraction = json.loads(json_content)
            
            # Save extracted JSON
            condition_part = condition if condition else "unknown_condition"
            output_file = f"results/extractions/{dataset_name}/{condition_part}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(extraction, f, indent=2)
            
            return True, extraction, None
        else:
            return False, None, "No JSON block found in response"
    
    except Exception as e:
        return False, None, f"LLM error: {str(e)}"


def run_extraction(dataset_name, instance_indices, condition=None):
    """Run extraction for a specific dataset and instances.
    
    Args:
        dataset_name: "credit", "law", "saudi", or "student"
        instance_indices: List of instance indices to process
        condition: Narrative condition folder ("include_pa", "exclude_pa", "override_pa/<label>", or None for all)
    """
    
    print("\n" + "=" * 100)
    print(f"EXTRACTION PIPELINE")
    print("=" * 100)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Narrative providers: {', '.join(NARRATIVE_PROVIDERS_TO_USE)}")
    print(f"Extractor providers: {', '.join(EXTRACTOR_PROVIDERS_TO_USE)}")
    print(f"Instances: {len(instance_indices)} total")
    if len(instance_indices) > 0:
        print(f"  Range: {min(instance_indices)}-{max(instance_indices)}")

    print(f"Total extractions: {len(NARRATIVE_PROVIDERS_TO_USE)} narrative providers × {len(EXTRACTOR_PROVIDERS_TO_USE)} extractor providers × {len(instance_indices)} instances = {len(NARRATIVE_PROVIDERS_TO_USE) * len(EXTRACTOR_PROVIDERS_TO_USE) * len(instance_indices)}")
    print("=" * 100)
    
    start_time = datetime.now()
    results = defaultdict(lambda: {"success": 0, "failed": 0, "errors": []})
    total_success = 0
    total_failed = 0
    
    total_extractions = len(NARRATIVE_PROVIDERS_TO_USE) * len(EXTRACTOR_PROVIDERS_TO_USE) * len(instance_indices)
    extraction_count = 0
    
    for instance_idx in instance_indices:
        for narrative_provider in NARRATIVE_PROVIDERS_TO_USE:
            for extractor_provider in EXTRACTOR_PROVIDERS_TO_USE:
                extraction_count += 1
                
                # Progress update every 20 extractions
                if extraction_count % 20 == 1:
                    pct = 100 * extraction_count // total_extractions
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if extraction_count > 1:
                        rate = elapsed / (extraction_count - 1)
                        remaining = (total_extractions - extraction_count) * rate
                        eta_str = f" - ETA: {int(remaining//60)}m {int(remaining%60)}s"
                    else:
                        eta_str = ""
                    print(f"Progress: {extraction_count}/{total_extractions} ({pct}%){eta_str}")
                    sys.stdout.flush()
                
                # Print what we're about to extract BEFORE attempting it
                print(f"  [{extraction_count}/{total_extractions}] Extracting instance {instance_idx} (narrative: {narrative_provider}, extractor: {extractor_provider})...", end=" ", flush=True)
                
                success, extraction, error = extract_single(
                    dataset_name=dataset_name,
                    instance_idx=instance_idx,
                    narrative_provider=narrative_provider,
                    extractor_provider=extractor_provider,
                    condition=condition
                )
                
                key = f"{narrative_provider}-{extractor_provider}"
                if success:
                    results[key]["success"] += 1
                    total_success += 1
                    print("✅")
                    sys.stdout.flush()
                else:
                    results[key]["failed"] += 1
                    results[key]["errors"].append(f"Instance {instance_idx}: {error}")
                    total_failed += 1
                    # Print error immediately so user can see what's failing
                    print(f"❌ {error}")
                    sys.stdout.flush()
    
    # Summary
    elapsed = datetime.now() - start_time
    total = total_success + total_failed
    
    print("\n" + "=" * 100)
    print(f"EXTRACTION COMPLETE")
    print(f"Total: {total_success + total_failed} extractions")
    print(f"  ✅ Successful: {total_success} ({100*total_success//total if total > 0 else 0}%)")
    print(f"  ❌ Failed: {total_failed} ({100*total_failed//total if total > 0 else 0}%)")
    print(f"Time: {int(elapsed.total_seconds()//60)}m {int(elapsed.total_seconds()%60)}s")
    print("=" * 100)
    
    # Provider summary
    if results:
        print("\nProvider Combination Summary:")
        print("-" * 100)
        print(f"{'Narrative':<15} {'Extractor':<15} {'Success':<10} {'Failed':<10} {'Success %':<12}")
        print("-" * 100)
        for key in sorted(results.keys()):
            parts = key.split('-')
            narrative_prov = parts[0] if len(parts) > 0 else "unknown"
            extractor_prov = parts[1] if len(parts) > 1 else "unknown"
            stats = results[key]
            total_prov = stats["success"] + stats["failed"]
            pct = 100 * stats["success"] // total_prov if total_prov > 0 else 0
            print(f"{narrative_prov:<15} {extractor_prov:<15} {stats['success']:<10} {stats['failed']:<10} {pct}%")
        print("=" * 100)
    
    return dict(results)




def main():
    """Run extraction with configuration from top of file."""
    
    # Determine which datasets to process
    if isinstance(DATASETS_TO_PROCESS, str):
        datasets_to_run = [DATASETS_TO_PROCESS]
    else:
        datasets_to_run = DATASETS_TO_PROCESS
    
    # Validate configuration
    for ds in datasets_to_run:
        if ds not in DATASETS:
            print(f"Error: Invalid dataset '{ds}'. Must be one of: {', '.join(DATASETS.keys())}")
            return
    
    invalid_extractors = set(EXTRACTOR_PROVIDERS_TO_USE) - set(LLM_MODELS.keys())
    if invalid_extractors:
        print(f"Error: Invalid EXTRACTOR_PROVIDERS_TO_USE: {', '.join(invalid_extractors)}")
        print(f"Available: {', '.join(LLM_MODELS.keys())}")
        return
    
    invalid_providers = set(NARRATIVE_PROVIDERS_TO_USE) - set(LLM_MODELS.keys())
    if invalid_providers:
        print(f"Error: Invalid NARRATIVE_PROVIDERS_TO_USE: {', '.join(invalid_providers)}")
        return
    
    # Determine instance indices for each dataset
    instance_indices_to_use = []
    if INSTANCE_INDICES == "all":
        # Will be set per-dataset below
        instance_indices_to_use = None
    else:
        instance_indices_to_use = INSTANCE_INDICES
    
    # Validate instance indices for all datasets
    for ds in datasets_to_run:
        max_instances = DATASETS[ds]["num_instances"]
        if instance_indices_to_use is not None:
            invalid_indices = [i for i in instance_indices_to_use if i < 0 or i >= max_instances]
            if invalid_indices:
                print(f"Error: Invalid INSTANCE_INDICES for {ds}: {invalid_indices}")
                print(f"Valid range: 0-{max_instances-1}")
                return
    
    # Run extraction for each dataset
    print(f"\n{'='*100}")
    dataset_word = "DATASET" if len(datasets_to_run) == 1 else "DATASETS"
    print(f"STARTING EXTRACTION FOR {len(datasets_to_run)} {dataset_word}")
    print(f"Datasets: {', '.join(datasets_to_run)}")
    print(f"{'='*100}")
    
    overall_start = datetime.now()
    all_results = {}
    
    for dataset_idx, dataset_name in enumerate(datasets_to_run, 1):
        print(f"\n[{dataset_idx}/{len(datasets_to_run)}] Processing {dataset_name.upper()} dataset...")
        
        # Determine instances for this dataset
        if instance_indices_to_use is None:
            # Process all instances for this dataset
            num_instances = DATASETS[dataset_name]["num_instances"]
            indices = list(range(num_instances))
        else:
            indices = instance_indices_to_use
        
        # Run extraction for this dataset
        run_extraction(dataset_name, indices, condition=NARRATIVE_CONDITION)
        
        print(f"✅ Completed {dataset_name.upper()}")
    
    # Final summary
    overall_elapsed = datetime.now() - overall_start
    print(f"\n{'='*100}")
    print(f"ALL EXTRACTIONS COMPLETE")
    print(f"Datasets processed: {', '.join(datasets_to_run)}")
    print(f"Total time: {int(overall_elapsed.total_seconds()//60)}m {int(overall_elapsed.total_seconds()%60)}s")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
