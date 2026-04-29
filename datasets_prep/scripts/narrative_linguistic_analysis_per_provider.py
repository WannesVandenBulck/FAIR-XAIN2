"""
Narrative Linguistic Analysis PER PROVIDER: Compare narrative characteristics across demographic groups.

Computes for all narratives per provider:
- Sentence length (average words per sentence)
- Number of sentences
- Number of characters
- Number of words
- Type-token ratio (unique words / total words)
- Flesch-Kincaid grade level (readability)
- Proportion of sentences with causal connectives

Compares differences between:
- Men vs Women
- Young vs Old

For each provider separately.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import re
import glob

ADVERSE_PATH = "datasets_prep/data/credit_dataset/credit_adverse.csv"
NARRATIVES_PATH = "results/narratives"
OUTPUT_PATH = "results/narrative_analysis/linguistic_analysis_by_provider.xlsx"

PROVIDERS = ["claude", "deepseek", "gemini", "grok", "mistral", "openai"]

# Causal connectives to search for
CAUSAL_CONNECTIVES = [
    'because', 'therefore', 'thus', 'as a result', 'consequently',
    'since', 'due to', 'on account of', 'for this reason', 'as a consequence',
    'hence', 'so that', 'in order to'
]

def extract_narratives_per_provider():
    """Find all narrative files and extract text with metadata including provider."""
    narratives_by_provider = {p: [] for p in PROVIDERS}
    
    # Find all narrative JSON files - broader search
    pattern = "results/narratives/**/*.json"
    narrative_files = glob.glob(pattern, recursive=True)
    
    for filepath in narrative_files:
        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
                narrative_text = data.get('narrative', '')
                
                if narrative_text:
                    # Extract instance index from filename
                    try:
                        instance_idx = int(filepath.split('instance_')[-1].split('.')[0])
                        
                        # Determine provider from filepath (handle both forward and backslashes)
                        filepath_normalized = filepath.replace('\\', '/')
                        provider = None
                        for prov in PROVIDERS:
                            if f"/{prov}/" in filepath_normalized:
                                provider = prov
                                break
                        
                        if provider:
                            narratives_by_provider[provider].append({
                                'instance_idx': instance_idx,
                                'narrative': narrative_text,
                                'provider': provider,
                                'filepath': filepath
                            })
                    except:
                        pass
        except:
            pass
    
    return narratives_by_provider

def compute_linguistic_metrics(narrative_text):
    """Compute all linguistic metrics for a single narrative."""
    
    # Clean text
    text = narrative_text.strip()
    
    # Basic counts
    num_characters = len(text)
    words = text.split()
    num_words = len(words)
    unique_words = set(w.lower() for w in words)
    num_unique_words = len(unique_words)
    
    # Sentences (split by . ! ?)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)
    
    # Sentence length
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    
    # Type-token ratio
    type_token_ratio = num_unique_words / num_words if num_words > 0 else 0
    
    # Flesch-Kincaid Grade Level
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        previous_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        return count
    
    total_syllables = sum(count_syllables(w) for w in words)
    
    if num_words > 0 and num_sentences > 0:
        fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59
        fk_grade = max(0, fk_grade)
    else:
        fk_grade = 0
    
    # Causal connectives
    causal_pattern = r'\b(' + '|'.join(CAUSAL_CONNECTIVES) + r')\b'
    num_causal = len(re.findall(causal_pattern, text.lower()))
    prop_sentences_with_causal = num_causal / num_sentences if num_sentences > 0 else 0
    
    return {
        'num_characters': num_characters,
        'num_words': num_words,
        'num_sentences': num_sentences,
        'avg_sentence_length': avg_sentence_length,
        'type_token_ratio': type_token_ratio,
        'flesch_kincaid_grade': fk_grade,
        'prop_sentences_with_causal': prop_sentences_with_causal,
        'num_causal_connectives': num_causal
    }

# Load adverse data for demographics
adverse_df = pd.read_csv(ADVERSE_PATH, index_col=0)
adverse_df['age_group'] = adverse_df['age'].apply(lambda x: 'young' if x < 32 else 'old')
sex_map = {0: 'male', 1: 'female'}
adverse_df['sex_label'] = adverse_df['sex'].map(sex_map)

# Extract narratives per provider
print("="*120)
print("NARRATIVE LINGUISTIC ANALYSIS - PER PROVIDER")
print("="*120)
print()
print("Loading narratives...")

narratives_by_provider = extract_narratives_per_provider()

total_narratives = sum(len(v) for v in narratives_by_provider.values())
print(f"Found {total_narratives} narratives across {len(PROVIDERS)} providers")
print()

# Compute metrics for all narratives
all_results = []

for provider in PROVIDERS:
    for narr_data in narratives_by_provider[provider]:
        instance_idx = narr_data['instance_idx']
        narrative_text = narr_data['narrative']
        
        # Get demographics
        if instance_idx < len(adverse_df):
            sex = adverse_df.iloc[instance_idx]['sex_label']
            age_group = adverse_df.iloc[instance_idx]['age_group']
            
            # Compute metrics
            metrics = compute_linguistic_metrics(narrative_text)
            
            result = {
                'provider': provider,
                'instance_idx': instance_idx,
                'sex': sex,
                'age_group': age_group,
                **metrics
            }
            all_results.append(result)

results_df = pd.DataFrame(all_results)

print(f"Computed metrics for {len(results_df)} narratives")
print()

# Define features to analyze
sex_features = [
    'num_characters', 'num_words', 'num_sentences', 'avg_sentence_length',
    'type_token_ratio', 'flesch_kincaid_grade', 'prop_sentences_with_causal'
]

# Store results for Excel
all_sex_comparisons = []
all_age_comparisons = []

# ============================================================================
# PER-PROVIDER ANALYSIS
# ============================================================================

for provider in PROVIDERS:
    print(f"\n{'='*120}")
    print(f"PROVIDER: {provider.upper()}")
    print(f"{'='*120}\n")
    
    provider_df = results_df[results_df['provider'] == provider]
    
    if len(provider_df) == 0:
        print("  No narratives found for this provider")
        continue
    
    print(f"Narratives: {len(provider_df)}")
    print(f"  Males: {len(provider_df[provider_df['sex'] == 'male'])}")
    print(f"  Females: {len(provider_df[provider_df['sex'] == 'female'])}")
    print(f"  Young: {len(provider_df[provider_df['age_group'] == 'young'])}")
    print(f"  Old: {len(provider_df[provider_df['age_group'] == 'old'])}")
    print()
    
    # ========================================================================
    # SEX COMPARISON
    # ========================================================================
    print(f"SEX COMPARISON (Male vs Female):")
    print(f"{'Metric':<35} {'Male Mean':>12} {'Female Mean':>12} {'Diff':>12} {'p-value':>10} {'Sig':>5}")
    print("-" * 120)
    
    for feature in sex_features:
        male_data = provider_df[provider_df['sex'] == 'male'][feature]
        female_data = provider_df[provider_df['sex'] == 'female'][feature]
        
        if len(male_data) > 0 and len(female_data) > 0:
            male_mean = male_data.mean()
            female_mean = female_data.mean()
            diff = male_mean - female_mean
            
            statistic, p_value = stats.mannwhitneyu(male_data, female_data, alternative='two-sided')
            is_sig = "YES" if p_value < 0.05 else "NO"
            
            print(f"{feature:<35} {male_mean:>12.3f} {female_mean:>12.3f} {diff:>12.3f} {p_value:>10.4f} {is_sig:>5}")
            
            all_sex_comparisons.append({
                'Provider': provider.capitalize(),
                'Feature': feature,
                'Male Mean': male_mean,
                'Female Mean': female_mean,
                'Difference (M-F)': diff,
                'p-value': p_value,
                'Significant': is_sig
            })
    
    # ========================================================================
    # AGE COMPARISON
    # ========================================================================
    print(f"\nAGE COMPARISON (Young vs Old):")
    print(f"{'Metric':<35} {'Young Mean':>12} {'Old Mean':>12} {'Diff':>12} {'p-value':>10} {'Sig':>5}")
    print("-" * 120)
    
    for feature in sex_features:
        young_data = provider_df[provider_df['age_group'] == 'young'][feature]
        old_data = provider_df[provider_df['age_group'] == 'old'][feature]
        
        if len(young_data) > 0 and len(old_data) > 0:
            young_mean = young_data.mean()
            old_mean = old_data.mean()
            diff = old_mean - young_mean
            
            statistic, p_value = stats.mannwhitneyu(young_data, old_data, alternative='two-sided')
            is_sig = "YES" if p_value < 0.05 else "NO"
            
            print(f"{feature:<35} {young_mean:>12.3f} {old_mean:>12.3f} {diff:>12.3f} {p_value:>10.4f} {is_sig:>5}")
            
            all_age_comparisons.append({
                'Provider': provider.capitalize(),
                'Feature': feature,
                'Young Mean': young_mean,
                'Old Mean': old_mean,
                'Difference (Old-Young)': diff,
                'p-value': p_value,
                'Significant': is_sig
            })

# ============================================================================
# EXCEL OUTPUT
# ============================================================================
print(f"\n{'='*120}")
print("Creating Excel output...")
print(f"{'='*120}\n")

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    # Raw data
    results_df.to_excel(writer, sheet_name='All Narratives', index=False)
    
    # Sex comparison summary
    sex_summary_df = pd.DataFrame(all_sex_comparisons)
    sex_summary_df.to_excel(writer, sheet_name='Sex Comparison', index=False)
    
    # Age comparison summary
    age_summary_df = pd.DataFrame(all_age_comparisons)
    age_summary_df.to_excel(writer, sheet_name='Age Comparison', index=False)
    
    # Per-provider sheets
    for provider in PROVIDERS:
        provider_df = results_df[results_df['provider'] == provider]
        if len(provider_df) > 0:
            provider_df.to_excel(writer, sheet_name=f'{provider.capitalize()} Data', index=False)

print(f"Saved to: {OUTPUT_PATH}")
print()
print("="*120)
print("SUMMARY")
print("="*120)
print()

# Summary of significant findings
sig_sex = sex_summary_df[sex_summary_df['Significant'] == 'YES']
sig_age = age_summary_df[age_summary_df['Significant'] == 'YES']

print(f"Significant sex differences (p < 0.05): {len(sig_sex)}")
if len(sig_sex) > 0:
    print(sig_sex[['Provider', 'Feature', 'Difference (M-F)', 'p-value']].to_string(index=False))
print()

print(f"Significant age differences (p < 0.05): {len(sig_age)}")
if len(sig_age) > 0:
    print(sig_age[['Provider', 'Feature', 'Difference (Old-Young)', 'p-value']].to_string(index=False))
