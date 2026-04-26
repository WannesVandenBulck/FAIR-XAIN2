"""
Test script: Send extractor prompt to LLM and get CSV response.

Tests the extraction pipeline with a single instance to verify the LLM
can successfully extract features and return a completed CSV.

Usage:
    python scripts/test_extraction.py [--provider PROVIDER] [--dataset DATASET] [--instance INSTANCE]

Example:
    python scripts/test_extraction.py --provider openai
    python scripts/test_extraction.py --provider claude --dataset law --instance 1
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_extractor_prompt import generate_extractor_prompt
from llm_tools.other.llm_client import generate_text


def test_extraction(dataset_name="credit", instance_idx=0, provider="openai", prompt_type="shap"):
    """
    Test the extraction pipeline with a single LLM call.
    
    Args:
        dataset_name: "credit" or "law"
        instance_idx: Instance number to extract
        provider: "openai", "claude", "gemini", "grok", "deepseek", "mistral"
        prompt_type: "shap" or "cf"
    
    Returns:
        LLM response (should be CSV)
    """
    
    print(f"🔄 Testing extraction for {dataset_name} instance {instance_idx} with {provider}")
    print("=" * 80)
    
    # Generate the extractor prompt
    print(f"\n📝 Generating extractor prompt...")
    prompt = generate_extractor_prompt(dataset_name, instance_idx, provider, prompt_type)
    
    # Prepare messages for LLM
    messages = [
        {
            "role": "system",
            "content": "You are an expert data analyst. Your task is to extract information from narratives and fill in a CSV template. Return ONLY the completed CSV in a code block - no other text."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Call LLM
    print(f"✉️  Sending prompt to {provider.upper()}...")
    print()
    
    try:
        response = generate_text(
            messages=messages,
            provider=provider,
            model=None,  # Use default model for provider
            temperature=0,
            max_tokens=8192  # Increased for longer CSV
        )
        
        # Save raw response for debugging
        raw_response_file = f"results/extraction_raw_response_{dataset_name}_instance_{instance_idx}_{provider}.txt"
        os.makedirs("results", exist_ok=True)
        with open(raw_response_file, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"💾 Raw response saved to: {raw_response_file}")
        
        print("✅ LLM Response received!")
        print("=" * 80)
        print(response[:500])
        print("...\n[RESPONSE TRUNCATED]")
        print("=" * 80)
        
        # Try to extract CSV from response
        if "```csv" in response:
            print("\n📊 CSV block detected in response")
            csv_start = response.find("```csv") + 7
            csv_end = response.find("```", csv_start)
            if csv_end == -1:
                csv_end = len(response)
            csv_content = response[csv_start:csv_end].strip()
            
            # Save extracted CSV
            output_file = f"results/extraction_response_{dataset_name}_instance_{instance_idx}_{provider}.csv"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(csv_content)
            print(f"💾 CSV saved to: {output_file}")
            
            # Analyze CSV content
            lines = csv_content.split('\n')
            print(f"\n📋 CSV Analysis:")
            print(f"  Total lines: {len(lines)}")
            print(f"  Header columns: {len(lines[0].split(','))}")
            
            if len(lines) > 1:
                data_line = lines[1]
                data_cols = data_line.split(',')
                print(f"  Data columns: {len(data_cols)}")
                
                # Count filled cells
                filled = sum(1 for v in data_cols if v.strip() and v.strip() != "***")
                print(f"  Filled cells: {filled} / {len(data_cols)}")
                print(f"  Completion: {100 * filled / len(data_cols):.1f}%")
                
                print(f"\n  First 200 chars of data:\n  {data_line[:200]}...")
            else:
                print("  ⚠️  No data row found in CSV!")
            
            return response
        else:
            print("\n⚠️  No CSV block found in response.")
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
    parser.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
    parser.add_argument("--dataset", default="credit", help="Dataset (default: credit)")
    parser.add_argument("--instance", type=int, default=0, help="Instance index (default: 0)")
    parser.add_argument("--prompt-type", default="shap", help="Prompt type (default: shap)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(f"EXTRACTION TEST - {args.dataset.upper()} Dataset, Instance {args.instance}")
    print("=" * 80 + "\n")
    
    response = test_extraction(
        dataset_name=args.dataset,
        instance_idx=args.instance,
        provider=args.provider,
        prompt_type=args.prompt_type
    )
    
    print("\n✨ Test complete!")

