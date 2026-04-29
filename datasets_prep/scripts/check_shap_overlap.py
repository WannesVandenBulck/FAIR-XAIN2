"""
Check if significant features (people_liable, number_credits) are more often 
in top-3 SHAP for females vs males.
"""

import json
import pandas as pd
from pathlib import Path

ADVERSE_PATH = "datasets_prep/data/credit_dataset/credit_adverse.csv"
GT_PATH = "results/ground_truth/credit"

adverse_df = pd.read_csv(ADVERSE_PATH, index_col=0)

# Focus on the two significant features
features_to_check = ["people_liable", "number_credits"]

results = {}
for feature in features_to_check:
    male_in_shap = 0
    female_in_shap = 0
    total_male = 0
    total_female = 0
    
    for instance_idx in range(34):
        gt_file = Path(GT_PATH) / f"credit/instance_{instance_idx}.json"
        if gt_file.exists():
            with open(gt_file) as f:
                gt_data = json.load(f)
                shap_names = {f['name'] for f in gt_data['most_important_features']}
                
                sex_numeric = adverse_df.iloc[instance_idx]['sex']
                sex_label = 'male' if sex_numeric == 0 else 'female'
                
                if sex_label == 'male':
                    total_male += 1
                    if feature in shap_names:
                        male_in_shap += 1
                else:
                    total_female += 1
                    if feature in shap_names:
                        female_in_shap += 1
    
    male_pct = (male_in_shap / total_male * 100) if total_male > 0 else 0
    female_pct = (female_in_shap / total_female * 100) if total_female > 0 else 0
    
    results[feature] = {
        'male_in_shap': male_in_shap,
        'male_total': total_male,
        'male_pct': male_pct,
        'female_in_shap': female_in_shap,
        'female_total': total_female,
        'female_pct': female_pct,
        'diff': female_pct - male_pct
    }

print("="*100)
print("SHAP TOP-3 OVERLAP ANALYSIS: Are significant features already in top-3 SHAP?")
print("="*100)
print()

for feature, data in results.items():
    print(f"{feature}:")
    print(f"  Males with this in top-3 SHAP:   {data['male_in_shap']:2d}/{data['male_total']:2d} ({data['male_pct']:5.1f}%)")
    print(f"  Females with this in top-3 SHAP: {data['female_in_shap']:2d}/{data['female_total']:2d} ({data['female_pct']:5.1f}%)")
    print(f"  Difference (F-M):                 {data['diff']:+5.1f}pp")
    print()

print("="*100)
print("INTERPRETATION:")
print("="*100)
print("If females have MORE of this feature in their top-3 SHAP:")
print("  → We exclude it from female mention counts (because it's in SHAP top-3)")
print("  → But we COUNT it for males (because it's NOT in their top-3 SHAP)")
print("  → This would artificially CREATE the observed difference!")
print()

for feature, data in results.items():
    if data['diff'] > 0:
        print(f"⚠️  {feature}: Females have {data['diff']:.1f}pp MORE of this in SHAP top-3")
        print(f"   This COULD explain the observed male bias in feature mentions!")
