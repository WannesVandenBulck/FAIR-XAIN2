"""
Feature mention percentages by provider and sex (CORRECTED).
Uses provider-specific narrative counts as denominator.
Excludes protected attributes {sex, age, foreign_worker}.
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

# Protected attributes to EXCLUDE
protected_attrs = {"sex", "age", "foreign_worker"}

# All non-protected features
all_features = ['amount', 'credit_history', 'duration', 'employment_duration', 
                'housing', 'installment_rate', 'job', 'number_credits', 
                'other_debtors', 'other_installment_plans', 'people_liable', 'present_residence', 
                'property', 'purpose', 'savings', 'status', 'telephone']

providers = ['claude', 'deepseek', 'gemini', 'grok', 'mistral', 'openai']

# Counters: provider -> feature -> (male_count, female_count)
feature_counts = {}
for provider in providers:
    feature_counts[provider] = {}
    for f in all_features:
        feature_counts[provider][f] = {'male': 0, 'female': 0}

# Track narrative counts per provider per sex
narrative_counts = {}
for provider in providers:
    narrative_counts[provider] = {'male': 0, 'female': 0}

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
        
        # Count this narrative
        narrative_counts[provider][sex] += 1
        
        # Count features
        for feature in data.get('features', []):
            feature_name = feature['name']
            mentioned = feature.get('mentioned')
            
            if isinstance(mentioned, str):
                mentioned = int(mentioned)
            
            # Count if: mentioned AND not in SHAP top 3 AND not protected
            if mentioned == 1 and feature_name not in shap_feature_names and feature_name not in protected_attrs:
                feature_counts[provider][feature_name][sex] += 1

# Build output table with percentages (using provider-specific denominators)
results = []
for feature in all_features:
    row = {'feature': feature}
    
    # Add columns for each provider (M/F) as percentages
    for provider in providers:
        male_count = feature_counts[provider][feature]['male']
        female_count = feature_counts[provider][feature]['female']
        male_narr = narrative_counts[provider]['male']
        female_narr = narrative_counts[provider]['female']
        
        male_pct = (male_count / male_narr * 100) if male_narr > 0 else 0
        female_pct = (female_count / female_narr * 100) if female_narr > 0 else 0
        row[f'{provider}_M'] = male_pct
        row[f'{provider}_F'] = female_pct
    
    # Add total (M/F) as percentages
    total_m_count = sum(feature_counts[p][feature]['male'] for p in providers)
    total_f_count = sum(feature_counts[p][feature]['female'] for p in providers)
    total_m_narr = sum(narrative_counts[p]['male'] for p in providers)
    total_f_narr = sum(narrative_counts[p]['female'] for p in providers)
    
    total_m_pct = (total_m_count / total_m_narr * 100) if total_m_narr > 0 else 0
    total_f_pct = (total_f_count / total_f_narr * 100) if total_f_narr > 0 else 0
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
    total_m_count = sum(feature_counts[provider][f]['male'] for f in all_features)
    total_f_count = sum(feature_counts[provider][f]['female'] for f in all_features)
    male_narr = narrative_counts[provider]['male']
    female_narr = narrative_counts[provider]['female']
    
    totals[f'{provider}_M'] = (total_m_count / male_narr * 100) if male_narr > 0 else 0
    totals[f'{provider}_F'] = (total_f_count / female_narr * 100) if female_narr > 0 else 0

total_m_count = sum(feature_counts[p][f]['male'] for p in providers for f in all_features)
total_f_count = sum(feature_counts[p][f]['female'] for p in providers for f in all_features)
total_m_narr = sum(narrative_counts[p]['male'] for p in providers)
total_f_narr = sum(narrative_counts[p]['female'] for p in providers)

totals['TOTAL_M'] = (total_m_count / total_m_narr * 100) if total_m_narr > 0 else 0
totals['TOTAL_F'] = (total_f_count / total_f_narr * 100) if total_f_narr > 0 else 0

totals_df = pd.DataFrame([totals])
totals_df = totals_df[col_order]

print("="*150)
print("Feature Mention % (Non-SHAP, Non-Protected) by Provider and Sex - CORRECTED")
print("="*150)
print(f"\nNarrative counts per provider:")
for provider in providers:
    m = narrative_counts[provider]['male']
    f = narrative_counts[provider]['female']
    print(f"  {provider:<12}: Male={m:2d}, Female={f:2d}")
print(f"\n")

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
output_file = "results/shap_metrics/sex/feature_count_non_shap_by_provider_CORRECTED.xlsx"
Path("results/shap_metrics/sex").mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='By Feature', index=False)
    totals_df.to_excel(writer, sheet_name='By Feature', startrow=len(df)+2, index=False)

print(f"Saved to: {output_file}")
