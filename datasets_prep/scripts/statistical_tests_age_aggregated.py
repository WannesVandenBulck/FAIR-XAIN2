"""
Statistical tests for all features - AGE comparison aggregated across ALL providers.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ADVERSE_PATH = "datasets_prep/data/credit_dataset/credit_adverse.csv"
GT_PATH = "results/ground_truth/credit"
EXTRACTIONS_PATH = "results/extractions/majority"
OUTPUT_PATH = "results/shap_metrics/age/statistical_tests_feature_mentions_age_AGGREGATED.xlsx"

PROVIDERS = ["claude", "deepseek", "gemini", "grok", "mistral", "openai"]
PROTECTED_ATTRS = {"sex", "age", "foreign_worker"}
AGE_THRESHOLD = 32

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
for provider in PROVIDERS:
    for instance_idx in range(34):
        extraction_file = Path(EXTRACTIONS_PATH) / f"{provider}/instance_{instance_idx}.json"
        if extraction_file.exists():
            with open(extraction_file) as f:
                extraction_data = json.load(f)
                for feature_dict in extraction_data.get('features', []):
                    fname = feature_dict.get('name')
                    if fname not in PROTECTED_ATTRS:
                        all_features.add(fname)

all_features = sorted(list(all_features))

print("="*100)
print("MANN-WHITNEY U TESTS: Feature Mentions by Age (Young < 32 vs Old >= 32)")
print("AGGREGATED ACROSS ALL PROVIDERS")
print("="*100)
print()

# Collect feature data - all mentions (0 and 1)
feature_data = {}

for provider in PROVIDERS:
    for instance_idx in range(34):
        extraction_file = Path(EXTRACTIONS_PATH) / f"{provider}/instance_{instance_idx}.json"
        if extraction_file.exists():
            with open(extraction_file) as f:
                extraction_data = json.load(f)
                
                age_numeric = adverse_df.iloc[instance_idx]['age']
                age_label = 'young' if age_numeric < AGE_THRESHOLD else 'old'
                shap_names = shap_features_by_instance.get(instance_idx, set())
                
                for feature_dict in extraction_data.get('features', []):
                    fname = feature_dict.get('name')
                    mentioned = feature_dict.get('mentioned', 0)
                    
                    # Only process if NOT in protected attrs AND NOT in SHAP top-3
                    if fname not in PROTECTED_ATTRS and fname not in shap_names:
                        if fname not in feature_data:
                            feature_data[fname] = {'young': [], 'old': []}
                        
                        # Append the mention value (0 or 1), not just when mentioned==1
                        if age_label == 'young':
                            feature_data[fname]['young'].append(mentioned)
                        else:
                            feature_data[fname]['old'].append(mentioned)

# Run Mann-Whitney U tests
results_list = []
bonferroni_threshold = 0.05 / len(feature_data)

for feature in sorted(feature_data.keys()):
    young_data = np.array(feature_data[feature]['young'], dtype=float)
    old_data = np.array(feature_data[feature]['old'], dtype=float)
    
    n_young = len(young_data)
    n_old = len(old_data)
    
    if n_young > 0 and n_old > 0:
        young_pct = (np.sum(young_data) / n_young * 100)
        old_pct = (np.sum(old_data) / n_old * 100)
        diff = old_pct - young_pct
        
        statistic, p_value = stats.mannwhitneyu(young_data, old_data, alternative='two-sided')
        
        is_sig = "YES" if p_value < 0.05 else "NO"
        is_bonf_sig = "YES" if p_value < bonferroni_threshold else "NO"
        
        results_list.append({
            'Feature': feature,
            'Young %': young_pct,
            'Old %': old_pct,
            'Diff (Old-Young)': diff,
            'p-value': p_value,
            'Significant (p<0.05)': is_sig,
            'Significant (Bonferroni)': is_bonf_sig,
            'N_Young': n_young,
            'N_Old': n_old
        })

# Sort by p-value
results_list.sort(key=lambda x: x['p-value'])

# Print results
print(f"{'Feature':<30} {'Young %':>10} {'Old %':>10} {'Diff':>10} {'p-value':>10} {'Sig':>5} {'N_Y':>5} {'N_O':>5}")
print("-" * 100)
for result in results_list:
    sig_marker = "YES" if result['p-value'] < 0.05 else "NO"
    print(f"{result['Feature']:<30} {result['Young %']:>9.1f}% {result['Old %']:>9.1f}% {result['Diff (Old-Young)']:>9.1f}% {result['p-value']:>10.4f} {sig_marker:>5} {result['N_Young']:>5} {result['N_Old']:>5}")

print()
print("="*100)
print(f"SIGNIFICANT FINDINGS (p < 0.05): {sum(1 for r in results_list if r['p-value'] < 0.05)} out of {len(results_list)} features")
print("="*100)
print()

for result in results_list:
    if result['p-value'] < 0.05:
        print(f"{result['Feature']}:")
        print(f"  Young: {result['Young %']:.1f}% of {result['N_Young']} narratives")
        print(f"  Old:   {result['Old %']:.1f}% of {result['N_Old']} narratives")
        print(f"  Difference: {result['Diff (Old-Young)']:+.1f}pp (Old {'favored' if result['Diff (Old-Young)'] > 0 else 'disfavored'})")
        print(f"  p-value: {result['p-value']:.4f}")
        print()

print("="*100)
print(f"BONFERRONI-CORRECTED SIGNIFICANCE (p < {bonferroni_threshold:.6f}): {sum(1 for r in results_list if r['p-value'] < bonferroni_threshold)} features")
print("="*100)
print()

for result in results_list:
    if result['p-value'] < bonferroni_threshold:
        print(f"{result['Feature']}: Young {result['Young %']:.1f}% vs Old {result['Old %']:.1f}%, p={result['p-value']:.4f}")

# Create Excel output
results_df = pd.DataFrame(results_list)
with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='All Results', index=False)
    
    # Significant findings sheet
    significant_df = results_df[results_df['p-value'] < 0.05]
    significant_df.to_excel(writer, sheet_name='Significant Only', index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
