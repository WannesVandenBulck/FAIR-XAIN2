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

# All features
all_features = ['amount', 'credit_history', 'duration', 'employment_duration', 
                'housing', 'installment_rate', 'job', 'number_credits', 
                'other_debtors', 'other_installment_plans', 'people_liable', 'present_residence', 
                'property', 'purpose', 'savings', 'status', 'telephone']

# Track which features mentioned per instance, age group
feature_details = {}

provider = 'gemini'
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
    
    # Track features for this instance
    for feature in data.get('features', []):
        feature_name = feature['name']
        mentioned = feature.get('mentioned', 0)
        
        if isinstance(mentioned, str):
            mentioned = int(mentioned)
        
        # Only track non-protected, non-SHAP features
        if feature_name not in protected_attrs and feature_name not in shap_feature_names:
            if mentioned == 1:
                if feature_name not in feature_details:
                    feature_details[feature_name] = {'young': [], 'old': []}
                feature_details[feature_name][age_group].append(instance_idx)

print("="*100)
print("GEMINI: FEATURE MENTIONS - DETAILED BREAKDOWN (excluding PROTECTED and SHAP)")
print("="*100)

print(f"\n{'Feature':<25} {'Young Count':>12} {'Young Inst':>20} {'Old Count':>12} {'Old Inst':>20}")
print("-"*100)

for feature in sorted(all_features):
    if feature in feature_details:
        young_count = len(feature_details[feature]['young'])
        old_count = len(feature_details[feature]['old'])
        young_insts = str(feature_details[feature]['young'])[:35]
        old_insts = str(feature_details[feature]['old'])[:35]
    else:
        young_count = 0
        old_count = 0
        young_insts = "[]"
        old_insts = "[]"
    
    marker = "***" if (young_count == 0 and old_count > 0) else ""
    print(f"{marker} {feature:<22} {young_count:>12} {young_insts:>20} {old_count:>12} {old_insts:>20}")

print("\n" + "="*100)
print("SUMMARY: Features EXCLUSIVE to old group (0 young, >0 old)")
print("="*100)

for feature in sorted(all_features):
    if feature in feature_details:
        young_count = len(feature_details[feature]['young'])
        old_count = len(feature_details[feature]['old'])
        if young_count == 0 and old_count > 0:
            print(f"  {feature:<25}: {old_count} instances (old)")
