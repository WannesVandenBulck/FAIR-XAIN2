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

# Protected attributes to EXCLUDE from main features count
protected_attrs = {"sex", "age", "foreign_worker"}

# All non-protected features (from SHAP analysis)
all_features = ['amount', 'credit_history', 'duration', 'employment_duration', 
                'housing', 'installment_rate', 'job', 'number_credits', 
                'other_debtors', 'other_installment_plans', 'people_liable', 'present_residence', 
                'property', 'purpose', 'savings', 'status', 'telephone']

# Counters for Gemini: feature -> age_group -> count
feature_mentions = {}
for f in all_features:
    feature_mentions[f] = {'young': 0, 'female': 0, 'old': 0}

# Track narrative counts per age group
narrative_counts = {'young': 0, 'old': 0}

# Process all instances for Gemini
provider = 'gemini'
for instance_idx in range(34):
    age_group = instance_demographics[instance_idx]['age_group']
    
    json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
    
    try:
        with open(json_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        continue
    
    # Count this narrative
    narrative_counts[age_group] += 1
    
    # Load ground truth to get SHAP features
    gt_file = Path(f"results/ground_truth/credit/instance_{instance_idx}.json")
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    shap_feature_names = {f['name'] for f in gt_data['most_important_features']}
    
    # Count features mentioned (excluding protected attrs AND SHAP features)
    for feature in data.get('features', []):
        feature_name = feature['name']
        mentioned = feature.get('mentioned')
        
        if isinstance(mentioned, str):
            mentioned = int(mentioned)
        
        # Count if: mentioned AND not in SHAP top 3 AND not protected
        if mentioned == 1 and feature_name not in shap_feature_names and feature_name not in protected_attrs:
            feature_mentions[feature_name][age_group] += 1

print("="*130)
print("GEMINI: FEATURE MENTIONS BY AGE GROUP (Excluding Protected Attributes & SHAP Top-3 Features)")
print("="*130)
print(f"\nNarrative counts: Young={narrative_counts['young']}, Old={narrative_counts['old']}\n")

# Build table with percentages and counts
results = []
for feature in all_features:
    young_count = feature_mentions[feature]['young']
    old_count = feature_mentions[feature]['old']
    young_pct = (young_count / narrative_counts['young'] * 100) if narrative_counts['young'] > 0 else 0
    old_pct = (old_count / narrative_counts['old'] * 100) if narrative_counts['old'] > 0 else 0
    diff = old_pct - young_pct
    
    results.append({
        'feature': feature,
        'young_count': young_count,
        'young_pct': young_pct,
        'old_count': old_count,
        'old_pct': old_pct,
        'diff': diff
    })

df = pd.DataFrame(results).sort_values('diff', ascending=False)

print(f"{'Feature':<25} {'Young Count':>12} {'Young %':>10} {'Old Count':>12} {'Old %':>10} {'Diff %':>10}")
print("-"*130)

for _, row in df.iterrows():
    marker = "***" if row['diff'] > 10 else "**" if row['diff'] > 5 else ""
    print(f"{marker} {row['feature']:<22} {int(row['young_count']):>12} {row['young_pct']:>9.1f}% {int(row['old_count']):>12} {row['old_pct']:>9.1f}% {row['diff']:>+9.1f}%")

print(f"\nFeatures mentioned MORE in OLD group (diff > 0):")
old_favored = df[df['diff'] > 0]
for _, row in old_favored.iterrows():
    print(f"  {row['feature']:<25}: {row['old_pct']:>6.1f}% (old) vs {row['young_pct']:>6.1f}% (young)")

print(f"\nFeatures mentioned MORE in YOUNG group (diff < 0):")
young_favored = df[df['diff'] < 0]
for _, row in young_favored.iterrows():
    print(f"  {row['feature']:<25}: {row['young_pct']:>6.1f}% (young) vs {row['old_pct']:>6.1f}% (old)")
