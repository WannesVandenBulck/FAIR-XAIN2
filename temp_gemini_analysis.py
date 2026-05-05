import pandas as pd
import numpy as np
from scipy import stats
import os
from glob import glob

# Load data
csv_files = glob("results/shap_metrics/age/per_instance_all_metrics_credit_*.csv")
latest_file = max(csv_files, key=os.path.getctime)
df = pd.read_csv(latest_file)

# Filter for Gemini only
gemini_df = df[df['provider'] == 'gemini']

# Separate by age group (excluding protected attributes and SHAP features)
young = gemini_df[gemini_df['age_group'] == 'young']
old = gemini_df[gemini_df['age_group'] == 'old']

print("="*100)
print("GEMINI: AGE GROUP COMPARISON (Young < 32 vs Old >= 32)")
print("="*100)
print(f"\nSample sizes: Young={len(young)}, Old={len(old)}\n")

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

print(f"{'Metric':<40} {'Young Mean':>12} {'Old Mean':>12} {'Diff':>12} {'p-value':>10} {'Sig':>5}")
print("-"*100)

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
    
    sig = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "ns"
    marker = "[SIG]" if p_mw < 0.05 else ""
    print(f"{marker} {metric:<40} {y_mean:>12.2f} {o_mean:>12.2f} {diff:>+12.2f} {p_mw:>10.4f} {sig:>5}")
