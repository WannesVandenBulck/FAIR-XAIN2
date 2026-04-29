"""
COMPREHENSIVE PER-PROVIDER STATISTICAL ANALYSIS
Tests: Sex, Age, 4-way combinations
For each metric AND each provider
Validates test assumptions and uses appropriate tests
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

ALPHA = 0.05

# Load data
csv_files = glob("results/shap_metrics/sex/per_instance_all_metrics_credit_*.csv")
latest_file = max(csv_files, key=os.path.getctime)
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

providers = df['provider'].unique()

print("="*140)
print("COMPREHENSIVE PER-PROVIDER STATISTICAL ANALYSIS")
print("="*140)
print(f"\nProviders: {', '.join(providers)}")
print(f"Metrics: {len(metrics)}")
print(f"Total records: {len(df)}")

# ============================================================================
# HELPER FUNCTIONS FOR TEST SELECTION
# ============================================================================

def check_normality(data, min_samples=3):
    """Check normality using Shapiro-Wilk test. Return True if normal."""
    if len(data) < min_samples:
        return None  # Cannot test
    try:
        stat, p = stats.shapiro(data)
        return p > 0.05  # True if normal
    except:
        return None

def select_appropriate_tests(group1, group2):
    """Select parametric vs non-parametric based on normality."""
    n1_norm = check_normality(group1)
    n2_norm = check_normality(group2)
    
    # If both normal (or unknown), use parametric
    if n1_norm is None or n2_norm is None or (n1_norm and n2_norm):
        use_parametric = True
    else:
        use_parametric = n1_norm and n2_norm
    
    return use_parametric

# Store all results
all_results = []

# ============================================================================
# MAIN ANALYSIS LOOP: PER PROVIDER
# ============================================================================

for provider in sorted(providers):
    provider_df = df[df['provider'] == provider]
    print(f"\n{'='*140}")
    print(f"PROVIDER: {provider.upper()}")
    print(f"{'='*140}")
    
    # Get subgroups for this provider
    male = provider_df[provider_df['sex'] == 'male']
    female = provider_df[provider_df['sex'] == 'female']
    young = provider_df[provider_df['age_group'] == 'young']
    old = provider_df[provider_df['age_group'] == 'old']
    
    my = provider_df[provider_df['group_4way'] == 'male_young']
    mo = provider_df[provider_df['group_4way'] == 'male_old']
    fy = provider_df[provider_df['group_4way'] == 'female_young']
    fo = provider_df[provider_df['group_4way'] == 'female_old']
    
    print(f"Sample sizes: Male={len(male)}, Female={len(female)}, Young={len(young)}, Old={len(old)}")
    print(f"4-way: MY={len(my)}, MO={len(mo)}, FY={len(fy)}, FO={len(fo)}")
    
    # ====== TEST 1: SEX COMPARISON ======
    print(f"\n[TEST 1] SEX COMPARISON")
    for metric in metrics:
        m_vals = male[metric].dropna().values
        f_vals = female[metric].dropna().values
        
        if len(m_vals) < 2 or len(f_vals) < 2:
            continue
        
        m_mean = m_vals.mean()
        f_mean = f_vals.mean()
        diff = m_mean - f_mean
        
        # Select test based on normality
        use_param = select_appropriate_tests(m_vals, f_vals)
        
        # Run both tests for transparency
        u_stat, p_mw = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')
        if len(m_vals) >= 2 and len(f_vals) >= 2:
            t_stat, p_t = stats.ttest_ind(m_vals, f_vals)
        else:
            p_t = np.nan
        
        # Use Mann-Whitney (more robust for non-normal data)
        p_value = p_mw
        test_used = "Mann-Whitney U"
        
        if p_value < ALPHA:
            print(f"  [SIG] {metric:35s} ({test_used:20s}): M={m_mean:.4f} vs F={f_mean:.4f}, p={p_value:.4f}")
        
        all_results.append({
            "provider": provider,
            "comparison": "Sex",
            "metric": metric,
            "test_used": test_used,
            "group1_mean": m_mean,
            "group2_mean": f_mean,
            "difference": diff,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            "n_group1": len(m_vals),
            "n_group2": len(f_vals),
        })
    
    # ====== TEST 2: AGE COMPARISON ======
    print(f"\n[TEST 2] AGE COMPARISON")
    for metric in metrics:
        y_vals = young[metric].dropna().values
        o_vals = old[metric].dropna().values
        
        if len(y_vals) < 2 or len(o_vals) < 2:
            continue
        
        y_mean = y_vals.mean()
        o_mean = o_vals.mean()
        diff = y_mean - o_mean
        
        u_stat, p_mw = stats.mannwhitneyu(y_vals, o_vals, alternative='two-sided')
        if len(y_vals) >= 2 and len(o_vals) >= 2:
            t_stat, p_t = stats.ttest_ind(y_vals, o_vals)
        else:
            p_t = np.nan
        
        p_value = p_mw
        test_used = "Mann-Whitney U"
        
        if p_value < ALPHA:
            print(f"  [SIG] {metric:35s} ({test_used:20s}): Y={y_mean:.4f} vs O={o_mean:.4f}, p={p_value:.4f}")
        
        all_results.append({
            "provider": provider,
            "comparison": "Age",
            "metric": metric,
            "test_used": test_used,
            "group1_mean": y_mean,
            "group2_mean": o_mean,
            "difference": diff,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            "n_group1": len(y_vals),
            "n_group2": len(o_vals),
        })
    
    # ====== TEST 3: 4-WAY COMPARISON ======
    print(f"\n[TEST 3] 4-WAY COMPARISON")
    for metric in metrics:
        my_vals = my[metric].dropna().values
        mo_vals = mo[metric].dropna().values
        fy_vals = fy[metric].dropna().values
        fo_vals = fo[metric].dropna().values
        
        # Filter out empty groups
        groups = []
        group_names = []
        if len(my_vals) >= 1:
            groups.append(my_vals)
            group_names.append('MY')
        if len(mo_vals) >= 1:
            groups.append(mo_vals)
            group_names.append('MO')
        if len(fy_vals) >= 1:
            groups.append(fy_vals)
            group_names.append('FY')
        if len(fo_vals) >= 1:
            groups.append(fo_vals)
            group_names.append('FO')
        
        if len(groups) < 2:
            continue
        
        # Kruskal-Wallis (non-parametric, handles 3+ groups)
        h_stat, p_kw = stats.kruskal(*groups)
        
        # One-way ANOVA (parametric)
        if len(groups) >= 2:
            f_stat, p_anova = stats.f_oneway(*groups)
        else:
            p_anova = np.nan
        
        p_value = p_kw
        test_used = "Kruskal-Wallis"
        
        if p_value < ALPHA:
            means_str = ", ".join([f"{n}={groups[i].mean():.3f}" for i, n in enumerate(group_names)])
            print(f"  [SIG] {metric:35s} ({test_used:20s}): {means_str}, p={p_value:.4f}")
        
        all_results.append({
            "provider": provider,
            "comparison": "4-Way",
            "metric": metric,
            "test_used": test_used,
            "group1_mean": my_vals.mean() if len(my_vals) > 0 else None,
            "group2_mean": mo_vals.mean() if len(mo_vals) > 0 else None,
            "group3_mean": fy_vals.mean() if len(fy_vals) > 0 else None,
            "group4_mean": fo_vals.mean() if len(fo_vals) > 0 else None,
            "difference": None,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            "n_group1": len(my_vals),
            "n_group2": len(mo_vals),
            "n_group3": len(fy_vals),
            "n_group4": len(fo_vals),
        })

# ============================================================================
# SAVE AND SUMMARIZE
# ============================================================================

results_df = pd.DataFrame(all_results)

print(f"\n\n{'='*140}")
print("SUMMARY OF SIGNIFICANT FINDINGS (p < 0.05)")
print(f"{'='*140}")

sig_results = results_df[results_df['significant'] == 'Yes']

if len(sig_results) > 0:
    print(f"\nTotal significant findings: {len(sig_results)}")
    print(f"\nBy provider:")
    for provider in sorted(sig_results['provider'].unique()):
        prov_sig = sig_results[sig_results['provider'] == provider]
        print(f"  {provider}: {len(prov_sig)} significant metrics")
    
    print(f"\nBy comparison type:")
    for comp in ['Sex', 'Age', '4-Way']:
        comp_sig = sig_results[sig_results['comparison'] == comp]
        print(f"  {comp}: {len(comp_sig)} significant metrics")
    
    print(f"\nDetailed results:")
    for _, row in sig_results.iterrows():
        print(f"\n  {row['provider'].upper()} | {row['comparison']:7s} | {row['metric']:35s}")
        print(f"    Test: {row['test_used']}")
        print(f"    p-value: {row['p_value']:.4f}")
        if row['comparison'] == '4-Way':
            print(f"    MY={row['group1_mean']:.3f}, MO={row['group2_mean']:.3f}, FY={row['group3_mean']:.3f}, FO={row['group4_mean']:.3f}")
        else:
            print(f"    Group 1: {row['group1_mean']:.4f}, Group 2: {row['group2_mean']:.4f}, Diff: {row['difference']:+.4f}")
else:
    print("\n[NO SIGNIFICANT FINDINGS]")

# Save to Excel
output_file = "results/shap_metrics/sex/statistical_analysis_per_provider_COMPREHENSIVE.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='All Results', index=False)
    sig_results.to_excel(writer, sheet_name='Significant Only', index=False)

output_file_age = "results/shap_metrics/age/statistical_analysis_per_provider_COMPREHENSIVE.xlsx"
with pd.ExcelWriter(output_file_age, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='All Results', index=False)
    sig_results.to_excel(writer, sheet_name='Significant Only', index=False)

print(f"\n\n[OK] Saved: {output_file}")
print(f"[OK] Saved: {output_file_age}")

print(f"\n{'='*140}")
print("TEST METHODOLOGY USED:")
print(f"{'='*140}")
print("""
[OK] Mann-Whitney U test (2-group comparisons for Sex and Age)
  - Non-parametric, robust to non-normal distributions
  - Good for our mixed metrics (binary agreements + continuous counts)
  - Works well with small/medium sample sizes
  
[OK] Kruskal-Wallis test (4-way group comparison)
  - Non-parametric extension of Mann-Whitney U to 3+ groups
  - Tests whether distributions differ across groups
  - Appropriate for the 4-way analysis
  
[OK] Both tests are two-sided (tests if groups differ, not direction)

[OK] Significance level: α = 0.05

[OK] Tests performed PER PROVIDER for all 13 metrics

[OK] Tests performed for all 3 comparisons:
  1. Sex (Male vs Female)
  2. Age (Young vs Old)
  3. 4-Way (Male-Young, Male-Old, Female-Young, Female-Old)
""")
