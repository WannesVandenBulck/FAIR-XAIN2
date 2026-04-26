#!/usr/bin/env python
"""
Quick viewer for extractor prompt in Markdown format

NOTE: The extraction prompt structure is IDENTICAL for all LLM providers.
The provider parameter only selects which narrative to display (since different
providers generate different narratives for the same instance).

Usage:
  python scripts/view_prompt_extract.py <instance_idx> [provider] [dataset]
  
Examples:
  python scripts/view_prompt_extract.py 0              # credit, instance 0, gemini narrative
  
python scripts/view_prompt_extract.py 0 claude        # Shows prompt with Claude's narrative
python scripts/view_prompt_extract.py 0 openai        # Shows same prompt with OpenAI's narrative
"""

from generate_extractor_prompt import generate_extractor_prompt
import sys
import io

# Handle Unicode encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Get arguments from command line
instance_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
provider = sys.argv[2] if len(sys.argv) > 2 else "gemini"
dataset_name = sys.argv[3] if len(sys.argv) > 3 else "credit"
prompt_type = "shap"  # Always SHAP for extractor

print(f"\n{'='*80}")
print(f"EXTRACTOR LLM PROMPT (Same for all LLMs)")
print(f"Dataset: {dataset_name.upper()} | Instance: {instance_idx} | Narrative from: {provider}")
print(f"{'='*80}\n")

prompt = generate_extractor_prompt(dataset_name, instance_idx, provider, prompt_type)
print(prompt)
