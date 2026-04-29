"""
Count feature mentions that are NOT in top SHAP features, separated by sex.
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# Load adverse_df to get demographic info
adverse_df = pd.read_csv("datasets_prep/data/credit_dataset/credit_adverse.csv")

# Map instance_idx to sex
instance_demographics = {}
for idx, row in adverse_df.iterrows():
    sex_numeric = row['sex']
    sex_label = 'male' if sex_numeric == 0 else 'female'
    instance_demographics[idx] = {'sex': sex_label}

# All features
all_features = ['age', 'amount', 'credit_history', 'duration', 'employment_duration', 
                'foreign_worker', 'housing', 'installment_rate', 'job', 'number_credits', 
                'other_debtors', 'other_installment_plans', 'people_liable', 'present_residence', 
                'property', 'purpose', 'savings', 'sex', 'status', 'telephone']

providers = ['claude', 'deepseek', 'gemini', 'grok', 'mistral', 'openai']

# Counters: feature -> provider -> {'male': count, 'female': count}
feature_counts = {}
for f in all_features:
    feature_counts[f] = {}
    for p in providers:
        feature_counts[f][p] = {'male': 0, 'female': 0}

# Process all instances and providers
for instance_idx in range(34):
    # Load ground truth to get SHAP features for this instance
    gt_file = Path(f"results/ground_truth/credit/instance_{instance_idx}.json")
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    shap_feature_names = {f['name'] for f in gt_data['most_important_features']}
    
    sex = instance_demographics[instance_idx]['sex']
    
    # Process each provider
    for provider in providers:
        json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
        
        try:
            with open(json_file) as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        
        # Count features
        for feature in data.get('features', []):
            feature_name = feature['name']
            mentioned = feature.get('mentioned')
            
            if isinstance(mentioned, str):
                mentioned = int(mentioned)
            
            # Count if: mentioned AND not in SHAP top 3
            if mentioned == 1 and feature_name not in shap_feature_names:
                feature_counts[feature_name][provider][sex] += 1

# Total narratives per sex
total_male_narratives = 126
total_female_narratives = 78

# Build output table with percentages
results = []
for feature in all_features:
    row = {'feature': feature}
    
    # Add columns for each provider (M/F) as percentages
    for provider in providers:
        male_pct = (feature_counts[feature][provider]['male'] / total_male_narratives) * 100
        female_pct = (feature_counts[feature][provider]['female'] / total_female_narratives) * 100
        row[f'{provider}_M'] = male_pct
        row[f'{provider}_F'] = female_pct
    
    # Add total (M/F) as percentages
    total_m = sum(feature_counts[feature][p]['male'] for p in providers)
    total_f = sum(feature_counts[feature][p]['female'] for p in providers)
    total_m_pct = (total_m / total_male_narratives) * 100
    total_f_pct = (total_f / total_female_narratives) * 100
    row['TOTAL_M'] = total_m_pct
    row['TOTAL_F'] = total_f_pct
    
    results.append(row)

df = pd.DataFrame(results)

# Sort by TOTAL_M descending
df = df.sort_values('TOTAL_M', ascending=False)

# Create nice column headers
col_order = ['feature']
for provider in providers:
    col_order.append(f'{provider}_M')
    col_order.append(f'{provider}_F')
col_order.extend(['TOTAL_M', 'TOTAL_F'])

df = df[col_order]

# Create totals row
totals = {'feature': 'TOTAL'}
for provider in providers:
    total_m_count = sum(feature_counts[f][provider]['male'] for f in all_features)
    total_f_count = sum(feature_counts[f][provider]['female'] for f in all_features)
    totals[f'{provider}_M'] = (total_m_count / total_male_narratives) * 100
    totals[f'{provider}_F'] = (total_f_count / total_female_narratives) * 100

total_m_count = sum(feature_counts[f][provider]['male'] for f in all_features for provider in providers)
total_f_count = sum(feature_counts[f][provider]['female'] for f in all_features for provider in providers)
totals['TOTAL_M'] = (total_m_count / total_male_narratives) * 100
totals['TOTAL_F'] = (total_f_count / total_female_narratives) * 100

totals_df = pd.DataFrame([totals])
totals_df = totals_df[col_order]

print("="*150)
print("Feature Mention Count (Non-SHAP) by Provider and Sex - As Percentage of Narratives")
print("="*150)
print(f"\nMale narratives: {total_male_narratives}, Female narratives: {total_female_narratives}\n")

# Format as percentages for display
df_display = df.copy()
for col in df_display.columns:
    if col != 'feature':
        df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%")

print(f"{df_display.to_string(index=False)}\n")

# Add totals row
totals_df_display = totals_df.copy()
for col in totals_df_display.columns:
    if col != 'feature':
        totals_df_display[col] = totals_df_display[col].apply(lambda x: f"{x:.1f}%")

print("="*150)
print(f"\n{totals_df_display.to_string(index=False)}\n")

# Save to Excel
output_file = "results/shap_metrics/sex/feature_count_non_shap_by_provider.xlsx"
Path("results/shap_metrics/sex").mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='By Feature', index=False)
    totals_df.to_excel(writer, sheet_name='By Feature', startrow=len(df)+2, index=False)

print(f"Saved to: {output_file}")
