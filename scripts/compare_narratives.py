"""
Compare all narratives for an instance across all providers, datasets, and prompt types.
Generates a readable comparison file to inspect differences between LLM outputs.
"""

import json
import os
from pathlib import Path

# Configuration
INSTANCE_IDX = 13
DATASETS = ["credit", "law"]
PROMPT_TYPES = ["shap", "cf"]
PROVIDERS = ["openai", "claude", "gemini", "grok", "deepseek", "mistral"]
BASE_PATH = Path("results/narratives")

def find_narrative_file(dataset, prompt_type, provider):
    """Find the narrative JSON file for a given dataset, prompt type, and provider."""
    provider_path = BASE_PATH / dataset / "narratives" / prompt_type / provider
    
    if not provider_path.exists():
        return None
    
    # Find the model subdirectory (there should be only one)
    for model_dir in provider_path.iterdir():
        if model_dir.is_dir():
            json_file = model_dir / f"instance_{INSTANCE_IDX}.json"
            if json_file.exists():
                return json_file
    
    return None

def get_narrative_content(json_file):
    """Extract narrative content from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        status = data.get("status", "unknown")
        if status == "success":
            return data.get("narrative", "[No narrative content]")
        else:
            error = data.get("error", "Unknown error")
            return f"[ERROR] {error}"
    except Exception as e:
        return f"[Failed to read file: {str(e)}]"

def main():
    output_file = Path(f"results/narratives/INSTANCE_{INSTANCE_IDX}_COMPARISON.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"# Instance {INSTANCE_IDX} - Narrative Comparison Across All Providers\n\n")
        out.write(f"Generated on: {__import__('datetime').datetime.now().isoformat()}\n\n")
        out.write(f"This file contains all narratives for instance {INSTANCE_IDX} from all 6 LLM providers,\n")
        out.write("across both datasets (credit, law) and both prompt types (SHAP, CF).\n\n")
        out.write("---\n\n")
        
        total = 0
        found = 0
        
        for dataset in DATASETS:
            out.write(f"\n## DATASET: {dataset.upper()}\n\n")
            
            for prompt_type in PROMPT_TYPES:
                out.write(f"### Prompt Type: {prompt_type.upper()}\n\n")
                
                for provider in PROVIDERS:
                    total += 1
                    json_file = find_narrative_file(dataset, prompt_type, provider)
                    
                    if json_file:
                        found += 1
                        model_dir = json_file.parent
                        model_name = model_dir.name
                        
                        out.write(f"#### Provider: {provider.upper()} (Model: {model_name})\n\n")
                        
                        narrative = get_narrative_content(json_file)
                        out.write(f"{narrative}\n\n")
                        out.write("---\n\n")
                    else:
                        out.write(f"#### Provider: {provider.upper()}\n\n")
                        out.write("[Narrative file not found]\n\n")
                        out.write("---\n\n")
    
    print(f"✓ Comparison file created: {output_file}")
    print(f"  Found {found}/{total} narratives")
    print(f"\nOpen this file to compare narratives across providers:")
    print(f"  {output_file.absolute()}")

if __name__ == "__main__":
    main()
