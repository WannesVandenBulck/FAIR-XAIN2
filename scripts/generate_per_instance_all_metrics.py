"""
Comprehensive: Generate per-instance metrics with ALL metrics for statistical testing.
Includes: features mentioned, feature values, protected attrs, rank/sign/value agreements, etc.
"""

import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

NUM_INSTANCES = 34
PROVIDERS = ["claude", "deepseek", "gemini", "grok", "mistral", "openai"]
DATASET_NAME = "credit"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_ground_truth(instance_idx):
    path = f"results/ground_truth/credit/instance_{instance_idx}.json"
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def load_extraction(provider, instance_idx):
    path = f"results/extractions/majority/{provider}/instance_{instance_idx}.json"
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def get_sex(gt):
    for feat in gt.get("features", []):
        if feat.get("name") == "sex":
            return feat.get("value")
    return None

def get_age(gt):
    for feat in gt.get("features", []):
        if feat.get("name") == "age":
            return feat.get("value")
    return None

def get_age_group(age_value):
    if age_value == "NaN" or age_value is None:
        return None
    try:
        age_int = int(age_value)
        return "young" if age_int < 32 else "old"
    except:
        return None

def calculate_metrics_comprehensive(extraction, ground_truth):
    """Calculate ALL metrics for one instance-provider combination."""
    
    protected_attrs = {"sex", "age", "foreign_worker"}
    shap_names = {f.get("name") for f in ground_truth.get("most_important_features", [])}
    
    # ====== FEATURES MENTIONED ======
    extracted_features = {f.get("name"): f for f in extraction.get("features", [])}
    gt_features = {f.get("name"): f for f in ground_truth.get("features", [])}
    
    # Count mentioned features
    features_mentioned = 0
    feature_values_given = 0
    for name in extracted_features:
        if name not in protected_attrs and name not in shap_names:
            features_mentioned += 1
            if extracted_features[name].get("mentioned", 0) == 1 and extracted_features[name].get("value") != "NaN":
                feature_values_given += 1
    
    # ====== PROTECTED ATTRIBUTES ======
    protected_mentioned = 0
    protected_values_given = 0
    for name in extracted_features:
        if name in protected_attrs:
            if extracted_features[name].get("mentioned", 0) == 1:
                protected_mentioned += 1
                if extracted_features[name].get("value") != "NaN":
                    protected_values_given += 1
    
    # ====== RANK AGREEMENTS (AS PERCENTAGES) ======
    extracted_shap_names = [f.get("name") for f in extraction.get("most_important_features", [])]
    gt_shap_names = [f.get("name") for f in ground_truth.get("most_important_features", [])]
    
    rank_agreements = []
    for i in range(min(len(extracted_shap_names), len(gt_shap_names))):
        rank_agreements.append(1 if extracted_shap_names[i] == gt_shap_names[i] else 0)
    
    rank_1 = (rank_agreements[0] * 100) if len(rank_agreements) > 0 else None
    rank_2 = (rank_agreements[1] * 100) if len(rank_agreements) > 1 else None
    rank_3 = (rank_agreements[2] * 100) if len(rank_agreements) > 2 else None
    rank_total = (np.mean(rank_agreements) * 100) if rank_agreements else None
    
    # ====== SIGN AGREEMENTS (BY NAME, AS PERCENTAGES) ======
    extracted_shap = {f.get("name"): f for f in extraction.get("most_important_features", [])}
    gt_shap = {f.get("name"): f for f in ground_truth.get("most_important_features", [])}
    
    sign_agreements = []
    for name in extracted_shap:
        if name in gt_shap:
            e_sign = extracted_shap[name].get("sign")
            g_sign = gt_shap[name].get("sign")
            if e_sign != "NaN" and g_sign != "NaN":
                sign_agreements.append(1 if e_sign == g_sign else 0)
    
    sign_agreement_mean = (np.mean(sign_agreements) * 100) if sign_agreements else None
    
    # ====== SHAP VALUE AGREEMENTS (BY NAME, AS PERCENTAGES) ======
    shap_value_agreements = []
    for name in extracted_shap:
        if name in gt_shap:
            e_val = extracted_shap[name].get("value")
            g_val = gt_shap[name].get("value")
            if e_val != "NaN" and g_val != "NaN" and e_val is not None and g_val is not None:
                shap_value_agreements.append(1 if e_val == g_val else 0)
    
    shap_value_agreement_mean = (np.mean(shap_value_agreements) * 100) if shap_value_agreements else None
    
    # ====== PROTECTED ATTR VALUE AGREEMENTS (AS PERCENTAGES) ======
    protected_value_agreements = []
    for name in extracted_features:
        if name in protected_attrs and name in gt_features:
            if extracted_features[name].get("mentioned", 0) == 1 and extracted_features[name].get("value") != "NaN":
                e_val = extracted_features[name].get("value")
                g_val = gt_features[name].get("value")
                if g_val != "NaN" and g_val is not None:
                    protected_value_agreements.append(1 if e_val == g_val else 0)
    
    protected_value_agreement_mean = (np.mean(protected_value_agreements) * 100) if protected_value_agreements else None
    
    # ====== OTHER FEATURES VALUE AGREEMENTS (AS PERCENTAGES) ======
    other_value_agreements = []
    for name in extracted_features:
        if name not in protected_attrs and name not in shap_names and name in gt_features:
            if extracted_features[name].get("mentioned", 0) == 1 and extracted_features[name].get("value") != "NaN":
                e_val = extracted_features[name].get("value")
                g_val = gt_features[name].get("value")
                if g_val != "NaN" and g_val is not None:
                    other_value_agreements.append(1 if e_val == g_val else 0)
    
    other_value_agreement_mean = (np.mean(other_value_agreements) * 100) if other_value_agreements else None
    
    # ====== ALL VALUE AGREEMENTS (AS PERCENTAGES) ======
    all_agreements = protected_value_agreements + other_value_agreements + shap_value_agreements
    all_value_agreement_mean = (np.mean(all_agreements) * 100) if all_agreements else None
    
    return {
        "features_mentioned": features_mentioned,
        "feature_values_given": feature_values_given,
        "protected_attrs_mentioned": protected_mentioned,
        "protected_attrs_values_given": protected_values_given,
        "rank_1_agreement": rank_1,
        "rank_2_agreement": rank_2,
        "rank_3_agreement": rank_3,
        "rank_total_agreement": rank_total,
        "sign_agreement_mean": sign_agreement_mean,
        "shap_value_agreement_mean": shap_value_agreement_mean,
        "protected_value_agreement_mean": protected_value_agreement_mean,
        "other_value_agreement_mean": other_value_agreement_mean,
        "all_value_agreement_mean": all_value_agreement_mean,
    }

# Collect all per-instance metrics
all_data = []

print("="*120)
print("GENERATING COMPREHENSIVE PER-INSTANCE METRICS")
print("="*120)

for instance_idx in range(NUM_INSTANCES):
    gt = load_ground_truth(instance_idx)
    if not gt:
        continue
    
    sex = get_sex(gt)
    age = get_age(gt)
    age_group = get_age_group(age)
    
    if sex is None or age_group is None:
        continue
    
    # Create group labels
    sex_label = "male" if sex == 0 else "female"
    group_4way = f"{sex_label}_{age_group}"  # e.g., "male_young"
    
    for provider in PROVIDERS:
        extraction = load_extraction(provider, instance_idx)
        if not extraction:
            continue
        
        metrics = calculate_metrics_comprehensive(extraction, gt)
        
        row = {
            "instance_idx": instance_idx,
            "provider": provider,
            "sex": sex_label,
            "age": age,
            "age_group": age_group,
            "group_4way": group_4way,
        }
        row.update(metrics)
        all_data.append(row)

df = pd.DataFrame(all_data)

print(f"\n{'='*120}")
print("DATA SUMMARY")
print(f"{'='*120}")
print(f"Total records: {len(df)}")
print(f"\nSex breakdown:")
print(f"  Male: {len(df[df['sex'] == 'male'])}")
print(f"  Female: {len(df[df['sex'] == 'female'])}")
print(f"\nAge group breakdown:")
print(f"  Young (< 32): {len(df[df['age_group'] == 'young'])}")
print(f"  Old (>= 32): {len(df[df['age_group'] == 'old'])}")
print(f"\n4-way groups:")
for group in df['group_4way'].unique():
    count = len(df[df['group_4way'] == group])
    print(f"  {group}: {count}")

print(f"\nSample metrics:")
print(df[['instance_idx', 'provider', 'sex', 'age_group', 'group_4way', 'rank_total_agreement', 
          'sign_agreement_mean', 'other_value_agreement_mean']].head(10).to_string())

# Save CSV
output_file = f"results/shap_metrics/sex/per_instance_all_metrics_credit_{TIMESTAMP}.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)
print(f"\n[OK] Saved: {output_file}")

# Also save to age folder
output_file_age = f"results/shap_metrics/age/per_instance_all_metrics_credit_{TIMESTAMP}.csv"
df.to_csv(output_file_age, index=False)
print(f"[OK] Saved: {output_file_age}")
