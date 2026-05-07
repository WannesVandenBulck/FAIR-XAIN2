"""Convert ground truth CSV files to JSON format matching extractor LLM template."""

import pandas as pd
import json
import os
from pathlib import Path


def convert_ground_truth_to_json(dataset_name="credit"):
    """
    Convert ground_truth CSV to JSON format matching extractor LLM template.
    
    JSON structure:
    {
      "predicted_probability": float or "NaN",
      "most_important_features": [
        {"rank": 1, "name": "feature_name", "sign": 1/-1, "value": float or "NaN"},
        ...
      ],
      "features": [
        {"name": "feature_name", "mentioned": 0/1, "value": float or "NaN"},
        ...
      ]
    }
    """
    
    # Load ground truth CSV
    gt_path = f"results/ground_truth_{dataset_name}.csv"
    df = pd.read_csv(gt_path)
    
    print(f"Loading {gt_path}")
    print(f"Found {len(df)} instances")
    
    # Determine number of SHAP features (3 for credit and law)
    num_shap_features = 3
    
    output_dir = f"results/ground_truth/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    converted_count = 0
    errors = []
    
    for idx, row in df.iterrows():
        try:
            instance_idx = int(row["instance_index"])
            
            # Build most_important_features from SHAP columns
            most_important_features = []
            for i in range(1, num_shap_features + 1):
                shap_name_col = f"SHAP_feature_{i}_name"
                shap_rank_col = f"SHAP_feature_{i}_rank"
                shap_sign_col = f"SHAP_feature_{i}_sign"
                shap_value_col = f"SHAP_feature_{i}_value"
                
                # Check if columns exist
                if all(col in row.index for col in [shap_name_col, shap_rank_col, shap_sign_col, shap_value_col]):
                    name = str(row[shap_name_col]) if pd.notna(row[shap_name_col]) else "NaN"
                    rank = int(row[shap_rank_col]) if pd.notna(row[shap_rank_col]) else i
                    sign = int(row[shap_sign_col]) if pd.notna(row[shap_sign_col]) else 0
                    value = row[shap_value_col]
                    
                    # Convert value to appropriate type
                    if pd.isna(value):
                        value = "NaN"
                    elif isinstance(value, (int, float)):
                        # Check if it's a whole number
                        if float(value) == int(value):
                            value = int(value)
                        else:
                            value = round(float(value), 2)
                    
                    most_important_features.append({
                        "rank": rank,
                        "name": name,
                        "sign": sign,
                        "value": value
                    })
            
            # Build features from other_feature columns
            features = []
            for i in range(17):  # 0-16 for credit (17 features)
                name_col = f"other_feature_{i}_name"
                value_col = f"other_feature_{i}_value"
                
                if all(col in row.index for col in [name_col, value_col]):
                    name = str(row[name_col]) if pd.notna(row[name_col]) else "NaN"
                    value = row[value_col]
                    
                    # Convert value to appropriate type
                    if pd.isna(value):
                        value = "NaN"
                    elif isinstance(value, (int, float)):
                        # Check if it's a whole number
                        if float(value) == int(value):
                            value = int(value)
                        else:
                            value = round(float(value), 2)
                    
                    # All features in ground truth are mentioned (1)
                    features.append({
                        "name": name,
                        "mentioned": 1,
                        "value": value
                    })
            
            # Build predicted probability
            predicted_prob = row.get("predicted_probability", "NaN")
            if pd.notna(predicted_prob):
                predicted_prob = round(float(predicted_prob), 2)
            else:
                predicted_prob = "NaN"
            
            # Build JSON object
            json_obj = {
                "predicted_probability": predicted_prob,
                "most_important_features": most_important_features,
                "features": features
            }
            
            # Save to JSON file
            output_file = f"{output_dir}/instance_{instance_idx}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2)
            
            converted_count += 1
            if (converted_count) % 10 == 0:
                print(f"  [OK] Converted {converted_count} instances")
        
        except Exception as e:
            error_msg = f"Instance {idx}: {str(e)}"
            errors.append(error_msg)
            print(f"  [ERROR] {error_msg}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"CONVERSION COMPLETE")
    print(f"Dataset: {dataset_name}")
    print(f"Successfully converted: {converted_count}/{len(df)} instances")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    return converted_count, errors


if __name__ == "__main__":
    # Convert credit dataset
    print("\n" + "=" * 80)
    print("CONVERTING GROUND TRUTH TO JSON FORMAT")
    print("=" * 80 + "\n")
    
    converted, errors = convert_ground_truth_to_json("credit")
    
    if not errors:
        print("\n[SUCCESS] All conversions successful!")
    else:
        print(f"\n⚠️  {len(errors)} errors occurred")
