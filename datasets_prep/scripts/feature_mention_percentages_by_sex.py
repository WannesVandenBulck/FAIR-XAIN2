"""
Calculate feature mention percentages by sex and age group.

For each provider (and aggregate), shows what % of male narratives mention each feature
vs what % of female narratives mention each feature.

Output: Tables showing feature coverage differences between demographic groups.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Load adverse_df to get demographic info
adverse_df = pd.read_csv("datasets_prep/data/credit_dataset/credit_adverse.csv")
print(f"Loaded adverse_df with {len(adverse_df)} rows")

# Map instance_idx to sex and age
# Sex encoding: 0 = male, 1 = female
instance_demographics = {}
for idx, row in adverse_df.iterrows():
    sex_numeric = row['sex']
    sex_label = 'male' if sex_numeric == 0 else 'female'
    instance_demographics[idx] = {
        'sex': sex_label,
        'age': row['age'],
        'age_group': 'young' if row['age'] < 32 else 'old'
    }

print(f"Mapped demographics for {len(instance_demographics)} instances")

# Providers
providers = ['claude', 'deepseek', 'gemini', 'grok', 'mistral', 'openai']

# Get all 20 credit features (from any majority_voted file)
sample_file = Path("results/extractions/majority/claude/instance_0.json")
with open(sample_file) as f:
    sample_data = json.load(f)
all_features = [f['name'] for f in sample_data['features']]
print(f"\nCredit features: {all_features}\n")

def calculate_feature_percentages_for_provider(provider):
    """Calculate feature mention % for males vs females for this provider."""
    
    # Counters for each sex
    male_mentions = defaultdict(int)
    female_mentions = defaultdict(int)
    male_total = 0
    female_total = 0
    
    # Process all 34 instances
    for instance_idx in range(34):
        # Get demographics
        sex = instance_demographics[instance_idx]['sex']
        
        # Load majority_voted JSON
        json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
        
        try:
            with open(json_file) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"WARNING: Missing {json_file}")
            continue
        
        # Count this narrative
        if sex == 'male':
            male_total += 1
        else:  # female
            female_total += 1
        
        # Process features
        for feature in data.get('features', []):
            feature_name = feature['name']
            mentioned = feature.get('mentioned')
            
            # Convert to int if string
            if isinstance(mentioned, str):
                mentioned = int(mentioned)
            
            if mentioned == 1:
                if sex == 'male':
                    male_mentions[feature_name] += 1
                else:
                    female_mentions[feature_name] += 1
    
    # Calculate percentages
    results = []
    for feature in all_features:
        male_pct = (male_mentions[feature] / male_total * 100) if male_total > 0 else 0
        female_pct = (female_mentions[feature] / female_total * 100) if female_total > 0 else 0
        diff = male_pct - female_pct
        
        results.append({
            'feature': feature,
            'male_pct': male_pct,
            'female_pct': female_pct,
            'diff': diff,
            'male_count': male_mentions[feature],
            'female_count': female_mentions[feature]
        })
    
    return pd.DataFrame(results), male_total, female_total

# ============================================================================
# Per-Provider Analysis
# ============================================================================

print("=" * 100)
print("PER-PROVIDER ANALYSIS: Feature Mention % by Sex")
print("=" * 100)

per_provider_dfs = {}

for provider in providers:
    df, male_total, female_total = calculate_feature_percentages_for_provider(provider)
    per_provider_dfs[provider] = (df, male_total, female_total)
    
    print(f"\n{provider.upper()}")
    print(f"  Male narratives: {male_total}, Female narratives: {female_total}")
    print("-" * 100)
    
    # Sort by difference
    df_sorted = df.sort_values('diff', ascending=False, key=abs)
    
    # Print header
    print(f"{'Feature':<25} {'Male %':>8} {'Female %':>10} {'Diff':>8} {'M Count':>8} {'F Count':>8}")
    print("-" * 100)
    
    # Print each feature
    for _, row in df_sorted.iterrows():
        print(f"{row['feature']:<25} {row['male_pct']:>7.1f}% {row['female_pct']:>9.1f}% {row['diff']:>7.1f}  {row['male_count']:>8.0f}  {row['female_count']:>8.0f}")

# ============================================================================
# Aggregate Analysis (All Providers Combined)
# ============================================================================

print("\n\n")
print("=" * 100)
print("AGGREGATE ANALYSIS: All Providers Combined")
print("=" * 100)

# Pool all narratives
agg_male_mentions = defaultdict(int)
agg_female_mentions = defaultdict(int)
agg_male_total = 0
agg_female_total = 0

for instance_idx in range(34):
    sex = instance_demographics[instance_idx]['sex']
    
    for provider in providers:
        json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
        
        try:
            with open(json_file) as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        
        # Count this narrative
        if sex == 'male':
            agg_male_total += 1
        else:  # female
            agg_female_total += 1
        
        # Process features
        for feature in data.get('features', []):
            feature_name = feature['name']
            mentioned = feature.get('mentioned')
            
            if isinstance(mentioned, str):
                mentioned = int(mentioned)
            
            if mentioned == 1:
                if sex == 'male':
                    agg_male_mentions[feature_name] += 1
                else:
                    agg_female_mentions[feature_name] += 1

# Calculate aggregate percentages
agg_results = []
for feature in all_features:
    male_pct = (agg_male_mentions[feature] / agg_male_total * 100) if agg_male_total > 0 else 0
    female_pct = (agg_female_mentions[feature] / agg_female_total * 100) if agg_female_total > 0 else 0
    diff = male_pct - female_pct
    
    agg_results.append({
        'feature': feature,
        'male_pct': male_pct,
        'female_pct': female_pct,
        'diff': diff,
        'male_count': agg_male_mentions[feature],
        'female_count': agg_female_mentions[feature]
    })

agg_df = pd.DataFrame(agg_results)

print(f"\nTotal: Male narratives: {agg_male_total}, Female narratives: {agg_female_total}")
print("-" * 100)

# Sort by difference
agg_sorted = agg_df.sort_values('diff', ascending=False, key=abs)

# Print header
print(f"{'Feature':<25} {'Male %':>8} {'Female %':>10} {'Diff':>8} {'M Count':>8} {'F Count':>8}")
print("-" * 100)

# Print each feature
for _, row in agg_sorted.iterrows():
    print(f"{row['feature']:<25} {row['male_pct']:>7.1f}% {row['female_pct']:>9.1f}% {row['diff']:>7.1f}  {row['male_count']:>8.0f}  {row['female_count']:>8.0f}")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n\n")
print("=" * 100)
print("SUMMARY: Features with Largest Percentage Point Differences")
print("=" * 100)

for provider in providers:
    df, _, _ = per_provider_dfs[provider]
    df_sorted = df.sort_values('diff', ascending=False, key=abs)
    
    top_3 = df_sorted.head(3)
    if len(top_3) > 0 and top_3.iloc[0]['diff'] != 0:
        print(f"\n{provider.upper()} - Top 3 disparities:")
        for _, row in top_3.iterrows():
            direction = "Male favored" if row['diff'] > 0 else "Female favored"
            print(f"  {row['feature']:<25}: {row['male_pct']:>6.1f}% vs {row['female_pct']:>6.1f}% (diff: {abs(row['diff']):>5.1f}pp) - {direction}")

print(f"\nAGGREGATE - Top 3 disparities:")
agg_sorted = agg_df.sort_values('diff', ascending=False, key=abs)
for _, row in agg_sorted.head(3).iterrows():
    direction = "Male favored" if row['diff'] > 0 else "Female favored"
    print(f"  {row['feature']:<25}: {row['male_pct']:>6.1f}% vs {row['female_pct']:>6.1f}% (diff: {abs(row['diff']):>5.1f}pp) - {direction}")

# ============================================================================
# Export to CSV
# ============================================================================

print("\n\nExporting to CSV files...")

# Per-provider CSVs
for provider in providers:
    df, _, _ = per_provider_dfs[provider]
    output_file = f"results/analysis/feature_mention_pct_{provider}_by_sex.csv"
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"  {output_file}")

# Aggregate CSV
agg_output_file = "results/analysis/feature_mention_pct_aggregate_by_sex.csv"
agg_df.to_csv(agg_output_file, index=False)
print(f"  {agg_output_file}")

print("\nDone!")
