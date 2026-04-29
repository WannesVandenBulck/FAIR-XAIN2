"""
Mann-Whitney U tests for feature mentions by sex, tested PER-PROVIDER.
Identifies which LLMs exhibit statistically significant feature mention biases across demographic groups.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path

# Paths
ADVERSE_PATH = "datasets_prep/data/credit_dataset/credit_adverse.csv"
GT_PATH = "results/ground_truth/credit"
EXTRACTIONS_PATH = "results/extractions/majority"
OUTPUT_PATH = "results/shap_metrics/sex/statistical_tests_per_provider.xlsx"

# Configuration
PROVIDERS = ["claude", "deepseek", "gemini", "grok", "mistral", "openai"]
PROTECTED_ATTRS = {"sex", "age", "foreign_worker"}

# Load adverse data for demographics
adverse_df = pd.read_csv(ADVERSE_PATH, index_col=0)

# Load ground truth SHAP features
shap_features_by_instance = {}
for instance_idx in range(34):
    gt_file = Path(GT_PATH) / f"credit/instance_{instance_idx}.json"
    if gt_file.exists():
        with open(gt_file) as f:
            gt_data = json.load(f)
            shap_features_by_instance[instance_idx] = {
                f['name'] for f in gt_data['most_important_features']
            }

# Collect feature data by provider
provider_results = {}

for provider in PROVIDERS:
    print(f"\n{'='*100}")
    print(f"PROVIDER: {provider.upper()}")
    print(f"{'='*100}\n")
    
    # Initialize feature data for this provider
    feature_data = {}
    for instance_idx in range(34):
        extraction_file = Path(EXTRACTIONS_PATH) / f"{provider}/instance_{instance_idx}.json"
        if extraction_file.exists():
            with open(extraction_file) as f:
                extraction_data = json.load(f)
                
                # Get sex from adverse_df
                sex_numeric = adverse_df.iloc[instance_idx]['sex']
                sex_label = 'male' if sex_numeric == 0 else 'female'
                
                # Get SHAP feature names for this instance
                shap_names = shap_features_by_instance.get(instance_idx, set())
                
                # Extract features
                for feature_dict in extraction_data.get('features', []):
                    feature_name = feature_dict.get('name')
                    mentioned = feature_dict.get('mentioned', 0)
                    
                    # Skip protected attributes and SHAP features
                    if (feature_name not in PROTECTED_ATTRS and 
                        feature_name not in shap_names and 
                        mentioned == 1):
                        
                        if feature_name not in feature_data:
                            feature_data[feature_name] = {'male': [], 'female': []}
                        feature_data[feature_name][sex_label].append(1)
    
    # Run Mann-Whitney U tests for this provider
    provider_test_results = []
    
    for feature in sorted(feature_data.keys()):
        male_data = feature_data[feature]['male']
        female_data = feature_data[feature]['female']
        
        n_male = len(male_data)
        n_female = len(female_data)
        male_pct = (sum(male_data) / n_male * 100) if n_male > 0 else 0
        female_pct = (sum(female_data) / n_female * 100) if n_female > 0 else 0
        diff = male_pct - female_pct
        
        # Perform Mann-Whitney U test
        if n_male > 0 and n_female > 0:
            statistic, p_value = stats.mannwhitneyu(
                np.array(male_data),
                np.array(female_data),
                alternative='two-sided'
            )
        else:
            p_value = 1.0
        
        provider_test_results.append({
            'Feature': feature,
            'Male %': male_pct,
            'Female %': female_pct,
            'Diff': diff,
            'p-value': p_value,
            'Significant (p<0.05)': 'YES' if p_value < 0.05 else 'NO',
            'N_Male': n_male,
            'N_Female': n_female
        })
    
    # Sort by p-value
    provider_test_results.sort(key=lambda x: x['p-value'])
    
    # Print results
    print(f"{'Feature':<30} {'Male %':>10} {'Female %':>10} {'Diff':>10} {'p-value':>10} {'Sig':>5} {'N_M':>5} {'N_F':>5}")
    print("-" * 100)
    for result in provider_test_results:
        sig_marker = "YES" if result['p-value'] < 0.05 else "NO"
        print(f"{result['Feature']:<30} {result['Male %']:>9.1f}% {result['Female %']:>9.1f}% {result['Diff']:>9.1f}% {result['p-value']:>10.4f} {sig_marker:>5} {result['N_Male']:>5} {result['N_Female']:>5}")
    
    significant_count = sum(1 for r in provider_test_results if r['p-value'] < 0.05)
    print(f"\nSignificant findings (p < 0.05): {significant_count} out of {len(provider_test_results)} features")
    
    provider_results[provider] = provider_test_results

# Create Excel output with per-provider sheets
with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    for provider in PROVIDERS:
        df = pd.DataFrame(provider_results[provider])
        df = df.sort_values('p-value')
        df.to_excel(writer, sheet_name=provider.capitalize(), index=False)
    
    # Create summary sheet
    summary_data = []
    for provider in PROVIDERS:
        significant = sum(1 for r in provider_results[provider] if r['p-value'] < 0.05)
        total_features = len(provider_results[provider])
        most_significant = min(provider_results[provider], key=lambda x: x['p-value'])
        summary_data.append({
            'Provider': provider.capitalize(),
            'Significant Features (p<0.05)': significant,
            'Total Features Tested': total_features,
            'Most Significant Feature': most_significant['Feature'],
            'Most Significant p-value': most_significant['p-value'],
            'Most Significant Diff (pp)': most_significant['Diff']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"\n{'='*100}")
print(f"Saved to: {OUTPUT_PATH}")
print(f"{'='*100}")
