"""Test script: Send extractor prompt to LLM and get JSON response.

Tests the extraction pipeline with a single instance to verify the LLM
can successfully extract features and return a completed JSON.

Usage:
    python scripts/test_extraction.py [--narrative-provider PROVIDER] [--extractor-provider PROVIDER] [--dataset DATASET] [--instance INSTANCE]

Example:
    python scripts/test_extraction.py --narrative-provider claude --extractor-provider openai
    python scripts/test_extraction.py --narrative-provider gemini --extractor-provider mistral --dataset law --instance 1
    python scripts/test_extraction.py --narrative-provider grok --extractor-provider claude --dataset credit --instance 0
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_extractor_prompt import generate_extractor_prompt
from llm_tools.other.llm_client import generate_text

# LLM provider configuration - matches make_narratives.py
LLM_MODELS = {
    "openai": "gpt-4o",
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-3-flash-preview",
    "grok": "grok-4-1-fast-non-reasoning",
    "deepseek": "deepseek-chat",
    "mistral": "mistral-large-latest",
}


def test_extraction(dataset_name="credit", instance_idx=0, narrative_provider="gemini", extractor_provider="openai", prompt_type="shap"):
    """
    Test the extraction pipeline with a single LLM call.
    
    Args:
        dataset_name: "credit" or "law"
        instance_idx: Instance number to extract
        narrative_provider: Provider whose narrative to use ("openai", "claude", "gemini", "grok", "deepseek", "mistral")
        extractor_provider: Provider to use for extraction (same options)
        prompt_type: "shap" or "cf"
    
    Returns:
        LLM response (should be JSON)
    """
    
    print(f"🔄 Testing extraction for {dataset_name} instance {instance_idx}")
    print(f"   Narrative source: {narrative_provider.upper()}")
    print(f"   Extractor LLM: {extractor_provider.upper()}")
    print("=" * 80)
    
    # Generate the extractor prompt (uses narrative_provider to select which narrative to read)
    print(f"\n📝 Generating extractor prompt with {narrative_provider} narrative...")
    prompt = generate_extractor_prompt(dataset_name, instance_idx, narrative_provider, prompt_type)
    
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
    
    # Call LLM for extraction
    print(f"✉️  Sending prompt to {extractor_provider.upper()} for extraction...")
    print()
    
    try:
        # Get the model for this provider
        if extractor_provider not in LLM_MODELS:
            raise ValueError(f"Unknown provider: {extractor_provider}. Available: {list(LLM_MODELS.keys())}")
        
        model = LLM_MODELS[extractor_provider]
        print(f"Using model: {model}")
        
        # For JSON extraction, we need fewer tokens than CSV
        # JSON response is compact; estimated ~1500 tokens max needed
        response = generate_text(
            messages=messages,
            provider=extractor_provider,
            model=model,
            temperature=0,
            max_tokens=2000
        )
        
        # Save raw response for debugging
        raw_response_file = f"results/extractions/{dataset_name}/raw/{prompt_type}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.txt"
        os.makedirs(os.path.dirname(raw_response_file), exist_ok=True)
        with open(raw_response_file, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"💾 Raw response saved to: {raw_response_file}")
        
        print("✅ LLM Response received!")
        print("=" * 80)
        print(response[:500])
        print("...\n[RESPONSE TRUNCATED]")
        print("=" * 80)
        
        # Try to extract JSON from response
        if "```json" in response:
            print("\n📊 JSON block detected in response")
            import json
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            if json_end == -1:
                json_end = len(response)
            json_content = response[json_start:json_end].strip()
            
            # Save extracted JSON
            output_file = f"results/extractions/{dataset_name}/extractions/{prompt_type}/{narrative_provider}/{extractor_provider}/instance_{instance_idx}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_content)
            print(f"💾 JSON saved to: {output_file}")
            
            # Analyze JSON content
            try:
                parsed = json.loads(json_content)
                print(f"\n📋 JSON Analysis:")
                print(f"  Top-level keys: {list(parsed.keys())}")
                
                # Count filled fields
                total_fields = 0
                filled_fields = 0
                
                # Count predicted_probability
                if "predicted_probability" in parsed:
                    total_fields += 1
                    val = parsed["predicted_probability"]
                    if val and val != "*":
                        filled_fields += 1
                    print(f"  predicted_probability: {val}")
                
                # Count most_important_features
                if "most_important_features" in parsed:
                    mif = parsed["most_important_features"]
                    print(f"  most_important_features: {len(mif)} items")
                    for i, item in enumerate(mif):
                        if i < 2:  # Display first 2
                            print(f"    [{i+1}] name={item.get('name')}, rank={item.get('rank')}, sign={item.get('sign')}, value={item.get('value')}")
                        total_fields += 4
                        if item.get('name') and item.get('name') != "*":
                            filled_fields += 1
                        if item.get('sign') and item.get('sign') != "*":
                            filled_fields += 1
                        if item.get('value') and item.get('value') != "*":
                            filled_fields += 1
                    if len(mif) > 2:
                        print(f"    ... and {len(mif) - 2} more")
                
                # Count features
                if "features" in parsed:
                    feats = parsed["features"]
                    print(f"  features: {len(feats)} items")
                    for i, item in enumerate(feats):
                        if i < 2:  # Display first 2
                            print(f"    [{i+1}] name={item.get('name')}, mentioned={item.get('mentioned')}, value={item.get('value')}")
                        total_fields += 3
                        if item.get('mentioned') and item.get('mentioned') != "*":
                            filled_fields += 1
                        if item.get('value') and item.get('value') != "*":
                            filled_fields += 1
                    if len(feats) > 2:
                        print(f"    ... and {len(feats) - 2} more")
                
                completion = (100 * filled_fields / total_fields) if total_fields > 0 else 0
                print(f"\n  Filled fields: {filled_fields} / {total_fields} ({completion:.1f}%)")
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Failed to parse JSON: {e}")
                print(f"  JSON content: {json_content[:200]}...")
            
            return response
        else:
            print("\n⚠️  No JSON block found in response.")
            print("Raw response (first 500 chars):")
            print(response[:500])
            return response
            
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test extraction pipeline with LLM")
    parser.add_argument("--narrative-provider", default="gemini", 
                        choices=list(LLM_MODELS.keys()),
                        help=f"Provider whose narrative to use (default: gemini). Available: {', '.join(LLM_MODELS.keys())}")
    parser.add_argument("--extractor-provider", default="openai", 
                        choices=list(LLM_MODELS.keys()),
                        help=f"Provider to use for extraction (default: openai). Available: {', '.join(LLM_MODELS.keys())}")
    parser.add_argument("--dataset", default="credit", choices=["credit", "law"],
                        help="Dataset (default: credit)")
    parser.add_argument("--instance", type=int, default=0, help="Instance index (default: 0)")
    parser.add_argument("--prompt-type", default="shap", choices=["shap", "cf"],
                        help="Prompt type (default: shap)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(f"EXTRACTION TEST - {args.dataset.upper()} Dataset, Instance {args.instance}")
    print(f"Narrative source: {args.narrative_provider.upper()} | Model: {LLM_MODELS[args.narrative_provider]}")
    print(f"Extractor LLM: {args.extractor_provider.upper()} | Model: {LLM_MODELS[args.extractor_provider]}")
    print("=" * 80 + "\n")
    
    response = test_extraction(
        dataset_name=args.dataset,
        instance_idx=args.instance,
        narrative_provider=args.narrative_provider,
        extractor_provider=args.extractor_provider,
        prompt_type=args.prompt_type
    )
    
    print("\n✨ Test complete!")

