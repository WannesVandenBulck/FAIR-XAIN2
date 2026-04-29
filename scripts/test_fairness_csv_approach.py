#!/usr/bin/env python
"""
Test to verify the CSV-based attribute injection works correctly.

Finds the threshold that results in ~50 adversely predicted instances,
then generates all 8 attribute combination batches.

Usage:
    python scripts/test_fairness_csv_approach.py
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fairness_evaluation import (
    _create_batch_adverse_csv,
    get_negatively_predicted_instances,
    get_attribute_combinations,
    encode_batch_id,
)


def test_csv_creation_all_batches():
    """Test that all 8 batch CSVs are created correctly with modified attributes."""
    
    dataset_name = "credit"
    batch_name = "fairness_test"
    threshold = 0.5
    
    # Get instances at threshold 0.5
    instance_indices = get_negatively_predicted_instances(dataset_name, threshold=threshold)
    
    print("\n" + "=" * 80)
    print(f"FAIRNESS EVALUATION - CSV BATCH GENERATION")
    print("=" * 80)
    print(f"Threshold: {threshold}")
    print(f"Instances: {len(instance_indices)}")
    print(f"✓ Selected instances for fairness evaluation")
    
    # Step 2: Get all 8 attribute combinations
    combos = get_attribute_combinations(dataset_name)
    
    print("\n" + "=" * 80)
    print(f"GENERATING {len(combos)} BATCH CSVs")
    print("=" * 80)
    
    for i, combo in enumerate(combos, 1):
        batch_id = encode_batch_id(combo)
        
        print(f"\n[{i}/{len(combos)}] 📊 Creating batch: {batch_id}")
        print(f"         Attributes: {combo}")
        
        csv_path = _create_batch_adverse_csv(
            dataset_name, instance_indices, combo, batch_name, batch_id
        )
        
        # Load and verify
        df = pd.read_csv(csv_path)
        
        print(f"         ✓ CSV created: {csv_path}")
        print(f"         ✓ Rows: {len(df)}")
        
        # Check attribute values in the batch CSV
        sex_codes = {"male": 0, "female": 1}
        fw_codes = {"yes": 1, "no": 2}
        
        # Validate attributes
        validations = []
        
        if "sex" in combo:
            expected_sex = sex_codes[combo["sex"]]
            actual_sex = df['sex'].unique()
            is_correct = len(actual_sex) == 1 and actual_sex[0] == expected_sex
            validations.append(("sex", expected_sex, actual_sex[0] if len(actual_sex) == 1 else actual_sex, is_correct))
        
        if "foreign_worker" in combo:
            expected_fw = fw_codes[combo["foreign_worker"]]
            actual_fw = df['foreign_worker'].unique()
            is_correct = len(actual_fw) == 1 and actual_fw[0] == expected_fw
            validations.append(("fw", expected_fw, actual_fw[0] if len(actual_fw) == 1 else actual_fw, is_correct))
        
        # Print validation results
        for attr, expected, actual, is_correct in validations:
            status = "✓" if is_correct else "✗"
            print(f"         {status} {attr}: expected={expected}, actual={actual}")
    
    print("\n" + "=" * 80)
    print(f"✅ ALL {len(combos)} BATCH CSVs CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nGenerated batches:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Threshold: {threshold}")
    print(f"  Instances per batch: {len(instance_indices)}")
    print(f"  Total batches: {len(combos)} (2 sex × 2 foreign_worker)")
    print(f"  Total narratives to generate: {len(instance_indices)} × 5 providers × {len(combos)} batches = {len(instance_indices) * 5 * len(combos)}")
    print(f"  Note: Age is a protected attribute but not modified - kept as real dataset values")
    print(f"\nCSVs location: results/fairness_eval/{batch_name}/data/")
    print(f"\nNext steps:")
    print(f"  1. Generate narratives: python scripts/fairness_cli.py generate --batch {batch_name} --threshold {threshold} --providers openai grok deepseek mistral gemini")
    print(f"  2. Extract narratives: python scripts/fairness_cli.py extract --batch {batch_name} --extractors openai grok deepseek mistral")
    print(f"  3. Compute metrics: python scripts/fairness_cli.py metrics --batch {batch_name}")


if __name__ == "__main__":
    test_csv_creation_all_batches()
