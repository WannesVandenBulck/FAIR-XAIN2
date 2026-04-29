"""
COMPREHENSIVE Statistical Significance Testing:
1. Sex (male vs female)
2. Age (young vs old)
3. 4-way combinations (male-young, male-old, female-young, female-old)
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
from glob import glob

ALPHA = 0.05

# Load data
csv_files = glob("results/shap_metrics/sex/per_instance_all_metrics_credit_*.csv")
if not csv_files:
    print("ERROR: No per-instance metrics CSV found")
    exit(1)

latest_file = max(csv_files, key=os.path.getctime)
print(f"Loading: {os.path.basename(latest_file)}\n")
df = pd.read_csv(latest_file)

# All metrics to test
metrics = [
    "features_mentioned",
    "feature_values_given", 
    "protected_attrs_mentioned",
    "protected_attrs_values_given",
    "rank_1_agreement",
    "rank_2_agreement",
    "rank_3_agreement",
    "rank_total_agreement",
    "sign_agreement_mean",
    "shap_value_agreement_mean",
    "protected_value_agreement_mean",
    "other_value_agreement_mean",
    "all_value_agreement_mean",
]

results_all = []

# ============================================================================
# TEST 1: SEX COMPARISON
# ============================================================================
print("="*120)
print("TEST 1: SEX COMPARISON (Male vs Female)")
print("="*120)

male = df[df['sex'] == 'male']
female = df[df['sex'] == 'female']

print(f"\nSample sizes: Male={len(male)}, Female={len(female)}\n")

sex_results = []
sig_sex_count = 0

for metric in metrics:
    m_vals = male[metric].dropna()
    f_vals = female[metric].dropna()
    
    if len(m_vals) < 2 or len(f_vals) < 2:
        continue
    
    m_mean = m_vals.mean()
    f_mean = f_vals.mean()
    m_std = m_vals.std()
    f_std = f_vals.std()
    diff = m_mean - f_mean
    
    # Tests
    u_stat, p_mw = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')
    t_stat, p_t = stats.ttest_ind(m_vals, f_vals)
    
    # Cohen's d
    pooled_std = np.sqrt(((len(m_vals)-1)*m_std**2 + (len(f_vals)-1)*f_std**2) / (len(m_vals) + len(f_vals) - 2))
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    sig = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else ""
    if p_mw < ALPHA:
        sig_sex_count += 1
        print(f"[SIG] {metric:35s}  M={m_mean:7.4f}  F={f_mean:7.4f}  Δ={diff:+.4f}  p={p_mw:.4f} {sig}")
    
    sex_results.append({
        "metric": metric,
        "male_mean": m_mean,
        "female_mean": f_mean,
        "male_std": m_std,
        "female_std": f_std,
        "diff": diff,
        "cohens_d": cohens_d,
        "p_mw": p_mw,
        "p_t": p_t,
        "n_male": len(m_vals),
        "n_female": len(f_vals)
    })

if sig_sex_count == 0:
    print("[NONE SIGNIFICANT]")

# ============================================================================
# TEST 2: AGE COMPARISON
# ============================================================================
print(f"\n{'='*120}")
print("TEST 2: AGE COMPARISON (Young < 32 vs Old >= 32)")
print(f"{'='*120}")

young = df[df['age_group'] == 'young']
old = df[df['age_group'] == 'old']

print(f"\nSample sizes: Young={len(young)}, Old={len(old)}\n")

age_results = []
sig_age_count = 0

for metric in metrics:
    y_vals = young[metric].dropna()
    o_vals = old[metric].dropna()
    
    if len(y_vals) < 2 or len(o_vals) < 2:
        continue
    
    y_mean = y_vals.mean()
    o_mean = o_vals.mean()
    y_std = y_vals.std()
    o_std = o_vals.std()
    diff = y_mean - o_mean
    
    u_stat, p_mw = stats.mannwhitneyu(y_vals, o_vals, alternative='two-sided')
    t_stat, p_t = stats.ttest_ind(y_vals, o_vals)
    
    pooled_std = np.sqrt(((len(y_vals)-1)*y_std**2 + (len(o_vals)-1)*o_std**2) / (len(y_vals) + len(o_vals) - 2))
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    sig = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else ""
    if p_mw < ALPHA:
        sig_age_count += 1
        print(f"[SIG] {metric:35s}  Y={y_mean:7.4f}  O={o_mean:7.4f}  Δ={diff:+.4f}  p={p_mw:.4f} {sig}")
    
    age_results.append({
        "metric": metric,
        "young_mean": y_mean,
        "old_mean": o_mean,
        "young_std": y_std,
        "old_std": o_std,
        "diff": diff,
        "cohens_d": cohens_d,
        "p_mw": p_mw,
        "p_t": p_t,
        "n_young": len(y_vals),
        "n_old": len(o_vals)
    })

if sig_age_count == 0:
    print("[NONE SIGNIFICANT]")

# ============================================================================
# TEST 3: 4-WAY COMPARISON
# ============================================================================
print(f"\n{'='*120}")
print("TEST 3: 4-WAY COMPARISON (Male-Young, Male-Old, Female-Young, Female-Old)")
print(f"{'='*120}")

groups_4way = {
    "male_young": df[df['group_4way'] == 'male_young'],
    "male_old": df[df['group_4way'] == 'male_old'],
    "female_young": df[df['group_4way'] == 'female_young'],
    "female_old": df[df['group_4way'] == 'female_old'],
}

print(f"\nSample sizes:")
for g, data in groups_4way.items():
    print(f"  {g}: {len(data)}")

print()

four_way_results = []
sig_4way_count = 0

for metric in metrics:
    # Extract values for each group
    group_vals = {}
    for g, data in groups_4way.items():
        vals = data[metric].dropna()
        if len(vals) >= 1:
            group_vals[g] = vals.values
    
    if len(group_vals) < 2:
        continue
    
    # Kruskal-Wallis test (non-parametric one-way)
    h_stat, p_kw = stats.kruskal(*[v for v in group_vals.values()])
    
    # One-way ANOVA (parametric)
    f_stat, p_anova = stats.f_oneway(*[v for v in group_vals.values()])
    
    sig = "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else ""
    if p_kw < ALPHA:
        sig_4way_count += 1
        means = {g: group_vals[g].mean() for g in group_vals}
        print(f"[SIG] {metric:35s}  p={p_kw:.4f} {sig}")
        for g in sorted(means.keys()):
            print(f"      {g:15s}: {means[g]:7.4f}")
    
    four_way_results.append({
        "metric": metric,
        "my_mean": group_vals.get('male_young', np.array([])).mean() if 'male_young' in group_vals else None,
        "mo_mean": group_vals.get('male_old', np.array([])).mean() if 'male_old' in group_vals else None,
        "fy_mean": group_vals.get('female_young', np.array([])).mean() if 'female_young' in group_vals else None,
        "fo_mean": group_vals.get('female_old', np.array([])).mean() if 'female_old' in group_vals else None,
        "p_kruskal_wallis": p_kw,
        "p_anova": p_anova,
    })

if sig_4way_count == 0:
    print("[NONE SIGNIFICANT]")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print(f"\n{'='*120}")
print("SAVING RESULTS TO EXCEL")
print(f"{'='*120}")

with pd.ExcelWriter("results/shap_metrics/sex/statistical_tests_COMPREHENSIVE.xlsx", engine='openpyxl') as writer:
    pd.DataFrame(sex_results).to_excel(writer, sheet_name='Sex', index=False)
    pd.DataFrame(age_results).to_excel(writer, sheet_name='Age', index=False)
    pd.DataFrame(four_way_results).to_excel(writer, sheet_name='4-Way', index=False)

with pd.ExcelWriter("results/shap_metrics/age/statistical_tests_COMPREHENSIVE.xlsx", engine='openpyxl') as writer:
    pd.DataFrame(sex_results).to_excel(writer, sheet_name='Sex', index=False)
    pd.DataFrame(age_results).to_excel(writer, sheet_name='Age', index=False)
    pd.DataFrame(four_way_results).to_excel(writer, sheet_name='4-Way', index=False)

print(f"\n[OK] results/shap_metrics/sex/statistical_tests_COMPREHENSIVE.xlsx")
print(f"[OK] results/shap_metrics/age/statistical_tests_COMPREHENSIVE.xlsx")

print(f"\n{'='*120}")
print("SUMMARY")
print(f"{'='*120}")
print(f"Significant differences found:")
print(f"  Sex comparison: {sig_sex_count} metrics")
print(f"  Age comparison: {sig_age_count} metrics")
print(f"  4-way comparison: {sig_4way_count} metrics")
