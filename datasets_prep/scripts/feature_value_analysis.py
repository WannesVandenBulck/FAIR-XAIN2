"""
Analysis: Do significantly different features have different VALUES across demographic groups?
If LLMs mention a feature more for Group A, is it because Group A actually has different values for that feature?

Significant features to check:
- By Sex: people_liable, number_credits (both favor males)
- By Age: people_liable, number_credits (favor old), housing (favors young)
"""

import pandas as pd
import numpy as np
from scipy import stats

ADVERSE_PATH = "datasets_prep/data/credit_dataset/credit_adverse.csv"

adverse_df = pd.read_csv(ADVERSE_PATH, index_col=0)

print("="*100)
print("FEATURE VALUE ANALYSIS: Are Mentioned Features Actually Different Across Groups?")
print("="*100)
print()

# ============================================================================
# SEX COMPARISON
# ============================================================================
print("="*100)
print("PART 1: SEX COMPARISON")
print("="*100)
print()

sex_labels = {0: 'male', 1: 'female'}
adverse_df['sex_label'] = adverse_df['sex'].map(sex_labels)
adverse_df['age_label'] = adverse_df['age'].apply(lambda x: 'young' if x < 32 else 'old')

# Features to check for sex bias
sex_features = ['people_liable', 'number_credits']

for feature in sex_features:
    print(f"\n{feature.upper()}")
    print("-" * 100)
    
    if feature in adverse_df.columns:
        male_values = adverse_df[adverse_df['sex'] == 0][feature]
        female_values = adverse_df[adverse_df['sex'] == 1][feature]
        
        print(f"Males (n={len(male_values)}):")
        print(f"  Mean: {male_values.mean():.2f}, Median: {male_values.median():.2f}, Std: {male_values.std():.2f}")
        print(f"  Range: [{male_values.min():.2f}, {male_values.max():.2f}]")
        
        print(f"Females (n={len(female_values)}):")
        print(f"  Mean: {female_values.mean():.2f}, Median: {female_values.median():.2f}, Std: {female_values.std():.2f}")
        print(f"  Range: [{female_values.min():.2f}, {female_values.max():.2f}]")
        
        print(f"Difference (M-F):")
        print(f"  Mean diff: {male_values.mean() - female_values.mean():.2f}")
        
        # Statistical test (Mann-Whitney U for non-parametric comparison)
        statistic, p_value = stats.mannwhitneyu(male_values, female_values, alternative='two-sided')
        print(f"  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  --> SIGNIFICANT VALUE DIFFERENCE (p < 0.05) ✓")
            print(f"      LLM bias might be EXPLAINED by data distribution")
        else:
            print(f"  --> NO significant value difference (p >= 0.05) ✗")
            print(f"      LLM bias is GENUINE feature selection bias (not data-driven)")
    else:
        print(f"  Feature not found in dataset")

# ============================================================================
# AGE COMPARISON
# ============================================================================
print("\n\n")
print("="*100)
print("PART 2: AGE COMPARISON")
print("="*100)
print()

age_features_old = ['people_liable', 'number_credits']
age_features_young = ['housing']

print("Features where OLD applicants get MORE mentions:")
for feature in age_features_old:
    print(f"\n{feature.upper()}")
    print("-" * 100)
    
    if feature in adverse_df.columns:
        young_values = adverse_df[adverse_df['age'] < 32][feature]
        old_values = adverse_df[adverse_df['age'] >= 32][feature]
        
        print(f"Young (n={len(young_values)}):")
        print(f"  Mean: {young_values.mean():.2f}, Median: {young_values.median():.2f}, Std: {young_values.std():.2f}")
        print(f"  Range: [{young_values.min():.2f}, {young_values.max():.2f}]")
        
        print(f"Old (n={len(old_values)}):")
        print(f"  Mean: {old_values.mean():.2f}, Median: {old_values.median():.2f}, Std: {old_values.std():.2f}")
        print(f"  Range: [{old_values.min():.2f}, {old_values.max():.2f}]")
        
        print(f"Difference (Old-Young):")
        print(f"  Mean diff: {old_values.mean() - young_values.mean():.2f}")
        
        statistic, p_value = stats.mannwhitneyu(young_values, old_values, alternative='two-sided')
        print(f"  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  --> SIGNIFICANT VALUE DIFFERENCE (p < 0.05) ✓")
            print(f"      LLM bias might be EXPLAINED by data distribution")
        else:
            print(f"  --> NO significant value difference (p >= 0.05) ✗")
            print(f"      LLM bias is GENUINE feature selection bias (not data-driven)")
    else:
        print(f"  Feature not found in dataset")

print("\n\nFeatures where YOUNG applicants get MORE mentions:")
for feature in age_features_young:
    print(f"\n{feature.upper()}")
    print("-" * 100)
    
    if feature in adverse_df.columns:
        young_values = adverse_df[adverse_df['age'] < 32][feature]
        old_values = adverse_df[adverse_df['age'] >= 32][feature]
        
        print(f"Young (n={len(young_values)}):")
        print(f"  Mean: {young_values.mean():.2f}, Median: {young_values.median():.2f}, Std: {young_values.std():.2f}")
        print(f"  Range: [{young_values.min():.2f}, {young_values.max():.2f}]")
        
        print(f"Old (n={len(old_values)}):")
        print(f"  Mean: {old_values.mean():.2f}, Median: {old_values.median():.2f}, Std: {old_values.std():.2f}")
        print(f"  Range: [{old_values.min():.2f}, {old_values.max():.2f}]")
        
        print(f"Difference (Old-Young):")
        print(f"  Mean diff: {old_values.mean() - young_values.mean():.2f}")
        
        statistic, p_value = stats.mannwhitneyu(young_values, old_values, alternative='two-sided')
        print(f"  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  --> SIGNIFICANT VALUE DIFFERENCE (p < 0.05) ✓")
            print(f"      LLM bias might be EXPLAINED by data distribution")
        else:
            print(f"  --> NO significant value difference (p >= 0.05) ✗")
            print(f"      LLM bias is GENUINE feature selection bias (not data-driven)")
    else:
        print(f"  Feature not found in dataset")

print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)
print()

summary_data = []

# Sex features
for feature in sex_features:
    if feature in adverse_df.columns:
        male_values = adverse_df[adverse_df['sex'] == 0][feature]
        female_values = adverse_df[adverse_df['sex'] == 1][feature]
        _, p_val = stats.mannwhitneyu(male_values, female_values, alternative='two-sided')
        
        summary_data.append({
            'Comparison': 'Sex (Male vs Female)',
            'Feature': feature,
            'LLM Bias': 'Males +13.3pp' if feature == 'people_liable' else 'Males +12.9pp',
            'Value Diff': f"{male_values.mean() - female_values.mean():.2f}",
            'p-value': f"{p_val:.4f}",
            'Explained?': 'YES' if p_val < 0.05 else 'NO'
        })

# Age features
for feature in age_features_old:
    if feature in adverse_df.columns:
        young_values = adverse_df[adverse_df['age'] < 32][feature]
        old_values = adverse_df[adverse_df['age'] >= 32][feature]
        _, p_val = stats.mannwhitneyu(young_values, old_values, alternative='two-sided')
        
        summary_data.append({
            'Comparison': 'Age (Old vs Young)',
            'Feature': feature,
            'LLM Bias': 'Old +17.6pp' if feature == 'people_liable' else 'Old +13.7pp',
            'Value Diff': f"{old_values.mean() - young_values.mean():.2f}",
            'p-value': f"{p_val:.4f}",
            'Explained?': 'YES' if p_val < 0.05 else 'NO'
        })

for feature in age_features_young:
    if feature in adverse_df.columns:
        young_values = adverse_df[adverse_df['age'] < 32][feature]
        old_values = adverse_df[adverse_df['age'] >= 32][feature]
        _, p_val = stats.mannwhitneyu(young_values, old_values, alternative='two-sided')
        
        summary_data.append({
            'Comparison': 'Age (Young vs Old)',
            'Feature': feature,
            'LLM Bias': 'Young +15.7pp',
            'Value Diff': f"{young_values.mean() - old_values.mean():.2f}",
            'p-value': f"{p_val:.4f}",
            'Explained?': 'YES' if p_val < 0.05 else 'NO'
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*100)
print("INTERPRETATION")
print("="*100)
print()
print("If Explained? = YES: Feature values ARE significantly different across groups")
print("  --> LLM might be responding to actual data patterns (not pure bias)")
print()
print("If Explained? = NO: Feature values are NOT significantly different across groups")
print("  --> LLM bias appears to be GENUINE stereotyping/bias (data-agnostic)")
