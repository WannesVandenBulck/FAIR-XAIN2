import json
from pathlib import Path

# Check young instances for Gemini
young_instances = [1, 3, 8, 9, 11, 14, 17, 19, 21, 22, 24, 25, 27, 29, 30, 32, 33]
old_instances = [0, 2, 4, 5, 6, 7, 10, 12, 13, 15, 16, 18, 20, 23, 26, 28, 31]

provider = 'gemini'

# Count credit_history and property OUTSIDE of SHAP
print("="*120)
print("GEMINI: credit_history and property mentions (EXCLUDING if in SHAP top-3)")
print("="*120)

young_ch_non_shap = 0
young_p_non_shap = 0
old_ch_non_shap = 0
old_p_non_shap = 0

print("\nYOUNG instances:")
print(f"{'Instance':<10} {'SHAP features':<40} {'ch mentioned':>15} {'ch in SHAP':>15} {'p mentioned':>15} {'p in SHAP':>15}")
print("-"*120)

for instance_idx in young_instances:
    gt_file = Path(f"results/ground_truth/credit/instance_{instance_idx}.json")
    with open(gt_file) as f:
        gt_data = json.load(f)
    shap_names = [f['name'] for f in gt_data['most_important_features']]
    
    json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
    with open(json_file) as f:
        data = json.load(f)
    
    ch_mentioned = False
    p_mentioned = False
    for feature in data.get('features', []):
        if feature['name'] == 'credit_history':
            ch_mentioned = feature.get('mentioned', 0) == 1
        if feature['name'] == 'property':
            p_mentioned = feature.get('mentioned', 0) == 1
    
    ch_in_shap = 'credit_history' in shap_names
    p_in_shap = 'property' in shap_names
    
    if ch_mentioned and not ch_in_shap:
        young_ch_non_shap += 1
    if p_mentioned and not p_in_shap:
        young_p_non_shap += 1
    
    ch_m = "YES" if ch_mentioned else "NO"
    p_m = "YES" if p_mentioned else "NO"
    ch_s = "YES" if ch_in_shap else "NO"
    p_s = "YES" if p_in_shap else "NO"
    
    print(f"{instance_idx:<10} {str(shap_names):<40} {ch_m:>15} {ch_s:>15} {p_m:>15} {p_s:>15}")

print("\n" + "="*120)
print("OLD instances:")
print(f"{'Instance':<10} {'SHAP features':<40} {'ch mentioned':>15} {'ch in SHAP':>15} {'p mentioned':>15} {'p in SHAP':>15}")
print("-"*120)

for instance_idx in old_instances:
    gt_file = Path(f"results/ground_truth/credit/instance_{instance_idx}.json")
    with open(gt_file) as f:
        gt_data = json.load(f)
    shap_names = [f['name'] for f in gt_data['most_important_features']]
    
    json_file = Path(f"results/extractions/majority/{provider}/instance_{instance_idx}.json")
    with open(json_file) as f:
        data = json.load(f)
    
    ch_mentioned = False
    p_mentioned = False
    for feature in data.get('features', []):
        if feature['name'] == 'credit_history':
            ch_mentioned = feature.get('mentioned', 0) == 1
        if feature['name'] == 'property':
            p_mentioned = feature.get('mentioned', 0) == 1
    
    ch_in_shap = 'credit_history' in shap_names
    p_in_shap = 'property' in shap_names
    
    if ch_mentioned and not ch_in_shap:
        old_ch_non_shap += 1
    if p_mentioned and not p_in_shap:
        old_p_non_shap += 1
    
    ch_m = "YES" if ch_mentioned else "NO"
    p_m = "YES" if p_mentioned else "NO"
    ch_s = "YES" if ch_in_shap else "NO"
    p_s = "YES" if p_in_shap else "NO"
    
    print(f"{instance_idx:<10} {str(shap_names):<40} {ch_m:>15} {ch_s:>15} {p_m:>15} {p_s:>15}")

print("\n" + "="*120)
print("SUMMARY (counting ONLY non-SHAP mentions):")
print("="*120)
print(f"credit_history - Young: {young_ch_non_shap}, Old: {old_ch_non_shap}")
print(f"property       - Young: {young_p_non_shap}, Old: {old_p_non_shap}")
