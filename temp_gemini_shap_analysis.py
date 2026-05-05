import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# Load adverse_df to get demographic info
adverse_df = pd.read_csv("datasets_prep/data/credit_dataset/credit_adverse.csv")

# Map instance_idx to age group
instance_demographics = {}
for idx, row in adverse_df.iterrows():
    age = row['age']
    age_group = 'young' if age < 32 else 'old'
    instance_demographics[idx] = {'age': age, 'age_group': age_group}

# Get SHAP top-3 feature names and their frequency by age group
shap_feature_frequency = {}

# Process all instances for Gemini
provider = 'gemini'
for instance_idx in range(34):
    age_group = instance_demographics[instance_idx]['age_group']
    
    # Load ground truth to get SHAP features
    gt_file = Path(f"results/ground_truth/credit/instance_{instance_idx}.json")
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    # Get top 3 SHAP features
    for idx, feature in enumerate(gt_data.get('most_important_features', [])[:3]):
        feature_name = feature['name']
        rank = idx + 1
        
        if feature_name not in shap_feature_frequency:
            shap_feature_frequency[feature_name] = {'young': [], 'old': []}
        
        shap_feature_frequency[feature_name][age_group].append(rank)

print("="*100)
print("GEMINI: TOP SHAP FEATURES BY AGE GROUP")
print("="*100)
print("\nTop features ranked by SHAP absolute values across instances:\n")

print(f"{'Feature':<25} {'Young (rank freq)':>25} {'Old (rank freq)':>25}")
print("-"*100)

for feature in sorted(shap_feature_frequency.keys()):
    young_ranks = shap_feature_frequency[feature]['young']
    old_ranks = shap_feature_frequency[feature]['old']
    
    young_freq = f"{len(young_ranks)} times (avg rank: {sum(young_ranks)/len(young_ranks):.1f})" if young_ranks else "Not in top 3"
    old_freq = f"{len(old_ranks)} times (avg rank: {sum(old_ranks)/len(old_ranks):.1f})" if old_ranks else "Not in top 3"
    
    print(f"{feature:<25} {young_freq:>25} {old_freq:>25}")

# Overall summary
print("\n" + "="*100)
print("SUMMARY: Average number of SHAP features mentioned by age group")
print("="*100)

# Calculate average features in SHAP top-3 by age group
young_total_features = 0
old_total_features = 0
young_count = 0
old_count = 0

for instance_idx in range(34):
    age_group = instance_demographics[instance_idx]['age_group']
    
    gt_file = Path(f"results/ground_truth/credit/instance_{instance_idx}.json")
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    num_features = len(gt_data.get('most_important_features', []))
    
    if age_group == 'young':
        young_total_features += num_features
        young_count += 1
    else:
        old_total_features += num_features
        old_count += 1

print(f"\nYoung group (< 32): Average {young_total_features/young_count:.2f} SHAP features per instance ({young_count} instances)")
print(f"Old group (>= 32):  Average {old_total_features/old_count:.2f} SHAP features per instance ({old_count} instances)")
