"""
Extraction script: Extract features and values from narratives using LLM.

Edit the CONFIGURATION section below to set:
- DATASET: which dataset to use (credit or law)
- NARRATIVE_PROVIDERS_TO_USE: which narrative providers to extract
- INSTANCE_INDICES: which instances to process
- EXTRACTOR_PROVIDER: which LLM to use for extraction

Then just run:
    python scripts/extraction.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json

# Add parent path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_extractor_prompt import generate_extractor_prompt
from llm_tools.other.llm_client import generate_text

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

DATASET = "credit"  # "credit" or "law"
NARRATIVE_PROVIDERS_TO_USE = ["gemini", "grok", "deepseek", "mistral", "openai", "claude"]  # All narrative providers: "gemini", "grok", "deepseek", "mistral", "openai", "claude"
INSTANCE_INDICES = list(range(34))  # All instances
EXTRACTOR_PROVIDER = "openai"  # LLM to use for extraction
PROMPT_TYPE = "shap"  # "shap" or "cf"

# ============================================================================

# LLM provider configuration
LLM_MODELS = {
    "openai": "gpt-4o",
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-3-flash-preview",
    "grok": "grok-4-1-fast-non-reasoning",
    "deepseek": "deepseek-chat",
    "mistral": "mistral-large-latest",
}

DATASETS = {
    "credit": {"num_instances": 34},
    "law": {"num_instances": 308}
}


def extract_single(dataset_name="credit", instance_idx=0, narrative_provider="gemini", 
                   extractor_provider="grok", prompt_type="shap"):
    """
    Extract features from a single narrative.
    
    Args:
        dataset_name: "credit" or "law"
        instance_idx: Instance number to extract
        narrative_provider: Provider whose narrative to use
        extractor_provider: Provider to use for extraction (default: grok)
        prompt_type: "shap" or "cf"
    
    Returns:
        (success: bool, extraction: dict or None, error: str or None)
    """
    
    # Generate the extractor prompt
    prompt = generate_extractor_prompt(dataset_name, instance_idx, narrative_provider, prompt_type)
    
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
    
    try:
        model = LLM_MODELS[extractor_provider]
        response = generate_text(
            messages=messages,
            provider=extractor_provider,
            model=model,
            temperature=0,
            max_tokens=2000
        )
        
        # Save raw response
        raw_response_file = f"results/extractions/{dataset_name}/raw/{prompt_type}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.txt"
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
            output_file = f"results/extractions/{dataset_name}/extractions/{prompt_type}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(extraction, f, indent=2)
            
            return True, extraction, None
        else:
            return False, None, "No JSON block found in response"
    
    except Exception as e:
        return False, None, f"LLM error: {str(e)}"


def run_extraction():
    """Run extraction based on configuration."""
    
    print("\n" + "=" * 100)
    print(f"EXTRACTION PIPELINE")
    print("=" * 100)
    print(f"Dataset: {DATASET.upper()}")
    print(f"Narrative providers: {', '.join(NARRATIVE_PROVIDERS_TO_USE)}")
    print(f"Instances: {len(INSTANCE_INDICES)} total")
    print(f"  Range: {min(INSTANCE_INDICES)}-{max(INSTANCE_INDICES)}")
    print(f"Extractor LLM: {EXTRACTOR_PROVIDER.upper()} ({LLM_MODELS[EXTRACTOR_PROVIDER]})")
    print(f"Prompt type: {PROMPT_TYPE.upper()}")
    print(f"Total extractions: {len(NARRATIVE_PROVIDERS_TO_USE)} providers × {len(INSTANCE_INDICES)} instances = {len(NARRATIVE_PROVIDERS_TO_USE) * len(INSTANCE_INDICES)}")
    print("=" * 100)
    
    start_time = datetime.now()
    results = defaultdict(lambda: {"success": 0, "failed": 0, "errors": []})
    total_success = 0
    total_failed = 0
    
    total_extractions = len(NARRATIVE_PROVIDERS_TO_USE) * len(INSTANCE_INDICES)
    extraction_count = 0
    
    for instance_idx in INSTANCE_INDICES:
        for narrative_provider in NARRATIVE_PROVIDERS_TO_USE:
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
            
            success, extraction, error = extract_single(
                dataset_name=DATASET,
                instance_idx=instance_idx,
                narrative_provider=narrative_provider,
                extractor_provider=EXTRACTOR_PROVIDER,
                prompt_type=PROMPT_TYPE
            )
            
            key = f"{narrative_provider}-{EXTRACTOR_PROVIDER}"
            if success:
                results[key]["success"] += 1
                total_success += 1
            else:
                results[key]["failed"] += 1
                results[key]["errors"].append(f"Instance {instance_idx}: {error}")
                total_failed += 1
    
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
        print("\nNarrative Provider Summary:")
        print("-" * 100)
        print(f"{'Narrative Provider':<20} {'Success':<10} {'Failed':<10} {'Success %':<12}")
        print("-" * 100)
        for key in sorted(results.keys()):
            narrative_prov = key.split('-')[0]
            stats = results[key]
            total_prov = stats["success"] + stats["failed"]
            pct = 100 * stats["success"] // total_prov if total_prov > 0 else 0
            print(f"{narrative_prov:<20} {stats['success']:<10} {stats['failed']:<10} {pct}%")
        print("=" * 100)
    
    return dict(results)




def main():
    """Run extraction with configuration from top of file."""
    
    # Validate configuration
    if DATASET not in DATASETS:
        print(f"Error: Invalid DATASET '{DATASET}'. Must be 'credit' or 'law'.")
        return
    
    if EXTRACTOR_PROVIDER not in LLM_MODELS:
        print(f"Error: Invalid EXTRACTOR_PROVIDER '{EXTRACTOR_PROVIDER}'.")
        print(f"Available: {', '.join(LLM_MODELS.keys())}")
        return
    
    if PROMPT_TYPE not in ["shap", "cf"]:
        print(f"Error: Invalid PROMPT_TYPE '{PROMPT_TYPE}'. Must be 'shap' or 'cf'.")
        return
    
    invalid_providers = set(NARRATIVE_PROVIDERS_TO_USE) - set(LLM_MODELS.keys())
    if invalid_providers:
        print(f"Error: Invalid NARRATIVE_PROVIDERS_TO_USE: {', '.join(invalid_providers)}")
        return
    
    max_instances = DATASETS[DATASET]["num_instances"]
    invalid_indices = [i for i in INSTANCE_INDICES if i < 0 or i >= max_instances]
    if invalid_indices:
        print(f"Error: Invalid INSTANCE_INDICES for {DATASET}: {invalid_indices}")
        print(f"Valid range: 0-{max_instances-1}")
        return
    
    # Run extraction
    run_extraction()


if __name__ == "__main__":
    main()
