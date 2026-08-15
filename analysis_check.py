import pandas as pd
import numpy as np

fc = pd.read_csv('results/fairness_eval/feature_coverage.csv')
faith = pd.read_csv('results/fairness_eval/faithfulness.csv')

print("="*90)
print("1. BIG DISPARITIES IN FEATURE COVERAGE (avg_other_mentioned)")
print("="*90)
print("\nComparing within demographic groups (same provider, different groups):")
print()

# Look for big differences in avg_other_mentioned within providers
for dataset in fc['dataset'].unique():
    for provider in fc['narrative_provider'].unique():
        subset = fc[(fc['dataset'] == dataset) & 
                    (fc['narrative_provider'] == provider) &
                    (fc['demographic_attribute'] != 'OVERALL')]
        
        if len(subset) > 0:
            max_other = subset['avg_other_mentioned'].max()
            min_other = subset['avg_other_mentioned'].min()
            diff = max_other - min_other
            
            if diff > 0.3:  # Threshold for "big" difference
                print(f"{dataset} - {provider}:")
                print(f"  Range in avg_other_mentioned: {min_other:.3f} to {max_other:.3f} (diff: {diff:.3f})")
                
                # Show which groups have the extremes
                max_row = subset.loc[subset['avg_other_mentioned'].idxmax()]
                min_row = subset.loc[subset['avg_other_mentioned'].idxmin()]
                print(f"  Max ({max_other:.3f}): {max_row['demographic_attribute']}={max_row['demographic_value']} (n={max_row['n_instances']})")
                print(f"  Min ({min_other:.3f}): {min_row['demographic_attribute']}={min_row['demographic_value']} (n={min_row['n_instances']})")
                print()

print("\n" + "="*90)
print("2. BIG DISPARITIES IN FAITHFULNESS METRICS")
print("="*90)
print("\nComparing within demographic groups (same provider, different groups):")
print()

# Check for big differences in other_value_accuracy
for dataset in faith['dataset'].unique():
    for provider in faith['narrative_provider'].unique():
        subset = faith[(faith['dataset'] == dataset) & 
                       (faith['narrative_provider'] == provider) &
                       (faith['demographic_attribute'] != 'OVERALL')]
        
        if len(subset) > 0:
            # Check other_value_accuracy (which had the big disparity you noticed)
            valid = subset['other_value_accuracy'].dropna()
            if len(valid) > 1:
                max_val = valid.max()
                min_val = valid.min()
                diff = max_val - min_val
                
                if diff > 0.1:  # Threshold
                    print(f"{dataset} - {provider} (other_value_accuracy):")
                    print(f"  Range: {min_val:.3f} to {max_val:.3f} (diff: {diff:.3f})")
                    max_row = subset.loc[valid.idxmax()]
                    min_row = subset.loc[valid.idxmin()]
                    print(f"  Max ({max_val:.3f}): {max_row['demographic_attribute']}={max_row['demographic_value']}")
                    print(f"  Min ({min_val:.3f}): {min_row['demographic_attribute']}={min_row['demographic_value']}")
                    print()

print("\n" + "="*90)
print("3. DIFFERENCES BETWEEN PROVIDERS (on same dataset)")
print("="*90)
print("\nOverall accuracy (OVERALL rows) by provider:")
print()

for dataset in faith['dataset'].unique():
    overall = faith[(faith['dataset'] == dataset) & (faith['demographic_value'] == 'ALL')]
    print(f"\n{dataset.upper()}:")
    print(overall[['narrative_provider', 'extractor_provider', 'rank_total_accuracy', 
                    'other_value_accuracy', 'all_value_accuracy']].to_string(index=False))

print("\n" + "="*90)
print("4. PROTECTED FEATURE MENTIONS (should be 0 for include_pa)")
print("="*90)
print()
protected = fc[(fc['avg_protected_mentioned'] > 0) & (fc['condition'] == 'include_pa')]
if len(protected) > 0:
    print("UNEXPECTED: Protected features mentioned in include_pa condition:")
    print(protected[['dataset', 'narrative_provider', 'demographic_attribute', 
                     'avg_protected_mentioned', 'avg_protected_values_given']].to_string(index=False))
else:
    print("✓ No protected features mentioned (as expected for include_pa)")

print("\n" + "="*90)
print("5. RANK ACCURACY DISPARITIES BY DEMOGRAPHIC GROUP")
print("="*90)
print()

for dataset in faith['dataset'].unique():
    for provider in faith['narrative_provider'].unique():
        subset = faith[(faith['dataset'] == dataset) & 
                       (faith['narrative_provider'] == provider) &
                       (faith['demographic_attribute'] != 'OVERALL')]
        
        if len(subset) > 0:
            # Check rank_total_accuracy
            max_rank = subset['rank_total_accuracy'].max()
            min_rank = subset['rank_total_accuracy'].min()
            diff = max_rank - min_rank
            
            if diff > 0.15:  # Threshold
                print(f"{dataset} - {provider} (rank_total_accuracy):")
                print(f"  Range: {min_rank:.3f} to {max_rank:.3f} (diff: {diff:.3f})")
                max_row = subset.loc[subset['rank_total_accuracy'].idxmax()]
                min_row = subset.loc[subset['rank_total_accuracy'].idxmin()]
                print(f"  Max ({max_rank:.3f}): {max_row['demographic_attribute']}={max_row['demographic_value']}")
                print(f"  Min ({min_rank:.3f}): {min_row['demographic_attribute']}={min_row['demographic_value']}")
                print()
