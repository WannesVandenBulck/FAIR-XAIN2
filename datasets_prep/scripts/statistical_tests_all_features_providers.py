"""
Statistical tests for all features across all providers.
Tests each feature for each provider individually to find provider-specific biases.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ADVERSE_PATH = "datasets_prep/data/credit_dataset/credit_adverse.csv"
GT_PATH = "results/ground_truth/credit"
EXTRACTIONS_PATH = "results/extractions/majority"
OUTPUT_PATH = "results/shap_metrics/sex/statistical_tests_all_features_all_providers.xlsx"

PROVIDERS = ["claude", "deepseek", "gemini", "grok", "mistral", "openai"]
PROTECTED_ATTRS = {"sex", "age", "foreign_worker"}

# Load adverse data
adverse_df = pd.read_csv(ADVERSE_PATH, index_col=0)

# Load SHAP features
shap_features_by_instance = {}
for instance_idx in range(34):
    gt_file = Path(GT_PATH) / f"instance_{instance_idx}.json"
    if gt_file.exists():
        with open(gt_file) as f:
            gt_data = json.load(f)
            shap_features_by_instance[instance_idx] = {
                f['name'] for f in gt_data['most_important_features']
            }

# Get all non-protected features
all_features = set()
for instance_idx in range(34):
    extraction_file = Path(EXTRACTIONS_PATH) / f"claude/instance_{instance_idx}.json"
    if extraction_file.exists():
        with open(extraction_file) as f:
            extraction_data = json.load(f)
            for feature_dict in extraction_data.get('features', []):
                fname = feature_dict.get('name')
                if fname not in PROTECTED_ATTRS:
                    all_features.add(fname)

all_features = sorted(list(all_features))

print("="*120)
print("STATISTICAL TESTS: All Features x All Providers")
print("="*120)
print()

# Store results for Excel output
all_provider_results = {}

for provider in PROVIDERS:
    print(f"\n{'='*120}")
    print(f"PROVIDER: {provider.upper()}")
    print(f"{'='*120}\n")
    
    provider_results = []
    
    for feature in all_features:
        male_data = []
        female_data = []
        
        for instance_idx in range(34):
            extraction_file = Path(EXTRACTIONS_PATH) / f"{provider}/instance_{instance_idx}.json"
            if extraction_file.exists():
                with open(extraction_file) as f:
                    extraction_data = json.load(f)
                    
                    sex_numeric = adverse_df.iloc[instance_idx]['sex']
                    sex_label = 'male' if sex_numeric == 0 else 'female'
                    shap_names = shap_features_by_instance.get(instance_idx, set())
                    
                    for feature_dict in extraction_data.get('features', []):
                        if feature_dict.get('name') == feature:
                            mentioned = feature_dict.get('mentioned', 0)
                            
                            if feature not in PROTECTED_ATTRS and feature not in shap_names:
                                if sex_label == 'male':
                                    male_data.append(mentioned)
                                else:
                                    female_data.append(mentioned)
                            break
        
        if len(male_data) > 0 and len(female_data) > 0:
            male_data = np.array(male_data, dtype=float)
            female_data = np.array(female_data, dtype=float)
            
            # Skip if either array is empty or all NaN
            if len(male_data) == 0 or len(female_data) == 0 or np.isnan(male_data).all() or np.isnan(female_data).all():
                continue
            
            statistic, p_value = stats.mannwhitneyu(male_data, female_data, alternative='two-sided')
            
            male_pct = (np.sum(male_data) / len(male_data) * 100)
            female_pct = (np.sum(female_data) / len(female_data) * 100)
            diff = female_pct - male_pct
            
            sig_marker = "YES" if p_value < 0.05 else "NO"
            
            provider_results.append({
                'Feature': feature,
                'Male %': male_pct,
                'Female %': female_pct,
                'Diff': diff,
                'p-value': p_value,
                'Significant (p<0.05)': sig_marker,
                'N_Male': len(male_data),
                'N_Female': len(female_data)
            })
    
    # Sort by p-value
    provider_results.sort(key=lambda x: x['p-value'])
    all_provider_results[provider] = provider_results
    
    # Print results for this provider
    print(f"{'Feature':<30} {'Male %':>10} {'Female %':>10} {'Diff':>10} {'p-value':>10} {'Sig':>5} {'N_M':>5} {'N_F':>5}")
    print("-" * 120)
    for result in provider_results:
        print(f"{result['Feature']:<30} {result['Male %']:>9.1f}% {result['Female %']:>9.1f}% {result['Diff']:>9.1f}% {result['p-value']:>10.4f} {result['Significant (p<0.05)']:>5} {result['N_Male']:>5} {result['N_Female']:>5}")
    
    significant_count = sum(1 for r in provider_results if r['p-value'] < 0.05)
    print(f"\nSignificant findings (p < 0.05): {significant_count} out of {len(provider_results)} features")

# Create Excel output
with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    # Individual provider sheets
    for provider in PROVIDERS:
        df = pd.DataFrame(all_provider_results[provider])
        df = df.sort_values('p-value')
        df.to_excel(writer, sheet_name=provider.capitalize(), index=False)
    
    # Summary sheet: aggregate findings
    summary_data = []
    for provider in PROVIDERS:
        significant_features = [r['Feature'] for r in all_provider_results[provider] if r['p-value'] < 0.05]
        summary_data.append({
            'Provider': provider.capitalize(),
            'Significant Features (p<0.05)': len(significant_features),
            'Total Features Tested': len(all_provider_results[provider]),
            'Features List': '; '.join(significant_features) if significant_features else 'None'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"\n{'='*120}")
print(f"Saved to: {OUTPUT_PATH}")
print(f"{'='*120}")
