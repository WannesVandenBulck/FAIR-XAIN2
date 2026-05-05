import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# Load adverse_df to get demographic info
adverse_df = pd.read_csv("datasets_prep/data/credit_dataset/credit_adverse.csv")

instance_demographics = {}
for idx, row in adverse_df.iterrows():
    age = row['age']
    age_group = 'young' if age < 32 else 'old'
    instance_demographics[idx] = {'age': age, 'age_group': age_group}

protected_attrs = {"sex", "age", "foreign_worker"}

# Process for Gemini
provider = 'gemini'
young_narratives = []
old_narratives = []

for instance_idx in range(34):
    age_group = instance_demographics[instance_idx]['age_group']
    
    json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
    
    try:
        with open(json_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        continue
    
    # Load ground truth to get SHAP features
    gt_file = Path(f"results/ground_truth/credit/instance_{instance_idx}.json")
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    shap_feature_names = {f['name'] for f in gt_data['most_important_features']}
    
    # Count features for this instance (excluding protected and SHAP)
    features_mentioned = 0
    features_with_values = 0
    
    for feature in data.get('features', []):
        feature_name = feature['name']
        mentioned = feature.get('mentioned', 0)
        value = feature.get('value')
        
        if isinstance(mentioned, str):
            mentioned = int(mentioned)
        
        # Count if: mentioned AND not protected AND not in SHAP
        if mentioned == 1 and feature_name not in protected_attrs and feature_name not in shap_feature_names:
            features_mentioned += 1
            if value != "NaN":
                features_with_values += 1
    
    if age_group == 'young':
        young_narratives.append({'instance': instance_idx, 'features': features_mentioned, 'values': features_with_values})
    else:
        old_narratives.append({'instance': instance_idx, 'features': features_mentioned, 'values': features_with_values})

young_df = pd.DataFrame(young_narratives)
old_df = pd.DataFrame(old_narratives)

print("="*100)
print("GEMINI: DETAILED FEATURE COUNTS PER INSTANCE (excluding PROTECTED and SHAP)")
print("="*100)
print("\nYOUNG (<32) narratives:")
print(young_df.to_string(index=False))
print(f"\nAverage: {young_df['features'].mean():.2f} features mentioned, {young_df['values'].mean():.2f} with values")

print("\n" + "="*100)
print("OLD (>=32) narratives:")
print(old_df.to_string(index=False))
print(f"\nAverage: {old_df['features'].mean():.2f} features mentioned, {old_df['values'].mean():.2f} with values")

print("\n" + "="*100)
print("SUMMARY:")
print("="*100)
print(f"Young: avg {young_df['features'].mean():.2f} features ± {young_df['features'].std():.2f}")
print(f"Old:   avg {old_df['features'].mean():.2f} features ± {old_df['features'].std():.2f}")
print(f"Difference: {old_df['features'].mean() - young_df['features'].mean():.2f} features")
