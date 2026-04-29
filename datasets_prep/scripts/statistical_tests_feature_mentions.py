"""
Statistical tests for feature mention differences by sex.
Tests each feature to find significant differences between male and female applicants.
Uses Mann-Whitney U test (non-parametric).
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from scipy import stats

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

# Collect mention data: feature -> sex -> [list of 0/1 mentions across all narratives]
feature_data = {}
for feature in all_features:
    feature_data[feature] = {'male': [], 'female': []}

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
        
        # Create a mention dict for this narrative
        features_dict = {f['name']: f for f in data.get('features', [])}
        
        # For each feature, record if it was mentioned
        for feature in all_features:
            if feature in features_dict:
                mentioned = features_dict[feature].get('mentioned')
                if isinstance(mentioned, str):
                    mentioned = int(mentioned)
            else:
                mentioned = 0
            
            # Only include if: not in SHAP top 3 AND not protected
            if feature not in shap_feature_names and feature not in protected_attrs:
                feature_data[feature][sex].append(mentioned)

# Run statistical tests
print("="*100)
print("MANN-WHITNEY U TESTS: Feature Mentions by Sex (Male vs Female)")
print("="*100)

results = []
for feature in all_features:
    male_data = np.array(feature_data[feature]['male'])
    female_data = np.array(feature_data[feature]['female'])
    
    # Skip if no data
    if len(male_data) == 0 or len(female_data) == 0:
        continue
    
    # Calculate summary stats
    male_mean = np.mean(male_data)
    female_mean = np.mean(female_data)
    male_std = np.std(male_data)
    female_std = np.std(female_data)
    
    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(male_data, female_data, alternative='two-sided')
    
    results.append({
        'feature': feature,
        'male_mean': male_mean,
        'female_mean': female_mean,
        'male_std': male_std,
        'female_std': female_std,
        'diff': male_mean - female_mean,
        'u_statistic': statistic,
        'p_value': p_value,
        'significant': 'YES' if p_value < 0.05 else 'NO',
        'n_male': len(male_data),
        'n_female': len(female_data)
    })

# Create DataFrame and sort by p-value
results_df = pd.DataFrame(results).sort_values('p_value')

print(f"\n{'Feature':<25} {'Male %':>8} {'Female %':>8} {'Diff':>8} {'p-value':>10} {'Sig':>5} {'N_M':>4} {'N_F':>4}")
print("-" * 100)

for _, row in results_df.iterrows():
    print(f"{row['feature']:<25} {row['male_mean']*100:>7.1f}% {row['female_mean']*100:>8.1f}% {row['diff']*100:>7.1f}% {row['p_value']:>10.4f} {row['significant']:>5} {int(row['n_male']):>4} {int(row['n_female']):>4}")

# Significant findings (p < 0.05)
sig_results = results_df[results_df['p_value'] < 0.05]

print("\n" + "="*100)
print(f"SIGNIFICANT FINDINGS (p < 0.05): {len(sig_results)} out of {len(results_df)} features")
print("="*100)

if len(sig_results) > 0:
    for _, row in sig_results.iterrows():
        direction = "Male" if row['diff'] > 0 else "Female"
        print(f"\n{row['feature']}:")
        print(f"  Male:   {row['male_mean']*100:.1f}% of {int(row['n_male'])} narratives")
        print(f"  Female: {row['female_mean']*100:.1f}% of {int(row['n_female'])} narratives")
        print(f"  Difference: {abs(row['diff'])*100:.1f}pp ({direction} favored)")
        print(f"  p-value: {row['p_value']:.4f}")
else:
    print("\nNo statistically significant differences found at p < 0.05.")

# Bonferroni correction for multiple comparisons
bonferroni_threshold = 0.05 / len(results_df)
sig_bonferroni = results_df[results_df['p_value'] < bonferroni_threshold]

print("\n" + "="*100)
print(f"BONFERRONI-CORRECTED SIGNIFICANCE (p < {bonferroni_threshold:.6f}): {len(sig_bonferroni)} features")
print("="*100)

if len(sig_bonferroni) > 0:
    for _, row in sig_bonferroni.iterrows():
        direction = "Male" if row['diff'] > 0 else "Female"
        print(f"\n{row['feature']}:")
        print(f"  Male:   {row['male_mean']*100:.1f}% vs Female: {row['female_mean']*100:.1f}%")
        print(f"  p-value: {row['p_value']:.4f} (HIGHLY SIGNIFICANT)")
else:
    print("\nNo features remain significant after Bonferroni correction.")

# Export to Excel
output_file = "results/shap_metrics/sex/statistical_tests_feature_mentions_CORRECTED.xlsx"
Path("results/shap_metrics/sex").mkdir(parents=True, exist_ok=True)

# Format for Excel
excel_df = results_df.copy()
excel_df['male_mean'] = excel_df['male_mean'].apply(lambda x: f"{x*100:.1f}%")
excel_df['female_mean'] = excel_df['female_mean'].apply(lambda x: f"{x*100:.1f}%")
excel_df['diff'] = excel_df['diff'].apply(lambda x: f"{x*100:.1f}%")
excel_df['p_value'] = excel_df['p_value'].apply(lambda x: f"{x:.6f}")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    excel_df[['feature', 'male_mean', 'female_mean', 'diff', 'p_value', 'significant', 'n_male', 'n_female']].to_excel(
        writer, sheet_name='All Results', index=False
    )
    sig_results_excel = sig_results.copy()
    sig_results_excel['male_mean'] = sig_results_excel['male_mean'].apply(lambda x: f"{x*100:.1f}%")
    sig_results_excel['female_mean'] = sig_results_excel['female_mean'].apply(lambda x: f"{x*100:.1f}%")
    sig_results_excel['diff'] = sig_results_excel['diff'].apply(lambda x: f"{x*100:.1f}%")
    sig_results_excel['p_value'] = sig_results_excel['p_value'].apply(lambda x: f"{x:.6f}")
    sig_results_excel[['feature', 'male_mean', 'female_mean', 'diff', 'p_value', 'n_male', 'n_female']].to_excel(
        writer, sheet_name='Significant Only', index=False
    )

print(f"\n\nSaved to: {output_file}")
