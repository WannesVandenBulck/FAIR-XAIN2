"""
Narrative Linguistic Analysis: Compare narrative characteristics across demographic groups.

Computes for all narratives:
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
OUTPUT_PATH = "results/narrative_analysis/linguistic_analysis_by_demographics.xlsx"

# Causal connectives to search for
CAUSAL_CONNECTIVES = [
    'because', 'therefore', 'thus', 'as a result', 'consequently',
    'since', 'due to', 'on account of', 'for this reason', 'as a consequence',
    'hence', 'so that', 'in order to'
]

def extract_narratives():
    """Find all narrative files and extract text with metadata."""
    narratives = []
    
    # Find all narrative JSON files
    pattern = str(Path(NARRATIVES_PATH) / "**" / "instance_*.json")
    narrative_files = glob.glob(pattern, recursive=True)
    
    for filepath in narrative_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
                narrative_text = data.get('narrative', '')
                
                if narrative_text:
                    # Extract metadata from path
                    parts = Path(filepath).parts
                    # narratives/credit/narratives/shap/claude/instance_0.json
                    try:
                        instance_idx = int(filepath.split('instance_')[-1].split('.')[0])
                        narratives.append({
                            'instance_idx': instance_idx,
                            'narrative': narrative_text,
                            'filepath': filepath
                        })
                    except:
                        pass
        except:
            pass
    
    return narratives

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
    # FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    # Simplified: count syllables by vowel groups
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
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
        # Minimum 1 syllable
        if count == 0:
            count = 1
        return count
    
    total_syllables = sum(count_syllables(w) for w in words)
    
    if num_words > 0 and num_sentences > 0:
        fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59
        fk_grade = max(0, fk_grade)  # Ensure non-negative
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

# Extract narratives
print("="*100)
print("NARRATIVE LINGUISTIC ANALYSIS")
print("="*100)
print()
print("Loading narratives...")

narratives = extract_narratives()
print(f"Found {len(narratives)} narratives")
print()

# Compute metrics for all narratives
results = []

for narr_data in narratives:
    instance_idx = narr_data['instance_idx']
    narrative_text = narr_data['narrative']
    
    # Get demographics
    if instance_idx < len(adverse_df):
        sex = adverse_df.iloc[instance_idx]['sex_label']
        age_group = adverse_df.iloc[instance_idx]['age_group']
        
        # Compute metrics
        metrics = compute_linguistic_metrics(narrative_text)
        
        result = {
            'instance_idx': instance_idx,
            'sex': sex,
            'age_group': age_group,
            **metrics
        }
        results.append(result)

results_df = pd.DataFrame(results)

print(f"Computed metrics for {len(results_df)} narratives")
print()

# ============================================================================
# SEX COMPARISON
# ============================================================================
print("="*100)
print("COMPARISON 1: SEX (Male vs Female)")
print("="*100)
print()

sex_features = [
    'num_characters', 'num_words', 'num_sentences', 'avg_sentence_length',
    'type_token_ratio', 'flesch_kincaid_grade', 'prop_sentences_with_causal'
]

print(f"{'Metric':<35} {'Male Mean':>12} {'Female Mean':>12} {'Diff':>12} {'p-value':>10} {'Sig':>5}")
print("-" * 100)

sex_results = []

for feature in sex_features:
    male_data = results_df[results_df['sex'] == 'male'][feature]
    female_data = results_df[results_df['sex'] == 'female'][feature]
    
    male_mean = male_data.mean()
    female_mean = female_data.mean()
    diff = male_mean - female_mean
    
    statistic, p_value = stats.mannwhitneyu(male_data, female_data, alternative='two-sided')
    is_sig = "YES" if p_value < 0.05 else "NO"
    
    print(f"{feature:<35} {male_mean:>12.3f} {female_mean:>12.3f} {diff:>12.3f} {p_value:>10.4f} {is_sig:>5}")
    
    sex_results.append({
        'Feature': feature,
        'Male Mean': male_mean,
        'Female Mean': female_mean,
        'Difference (M-F)': diff,
        'p-value': p_value,
        'Significant': is_sig
    })

# ============================================================================
# AGE COMPARISON
# ============================================================================
print("\n\n")
print("="*100)
print("COMPARISON 2: AGE (Young < 32 vs Old >= 32)")
print("="*100)
print()

print(f"{'Metric':<35} {'Young Mean':>12} {'Old Mean':>12} {'Diff':>12} {'p-value':>10} {'Sig':>5}")
print("-" * 100)

age_results = []

for feature in sex_features:
    young_data = results_df[results_df['age_group'] == 'young'][feature]
    old_data = results_df[results_df['age_group'] == 'old'][feature]
    
    young_mean = young_data.mean()
    old_mean = old_data.mean()
    diff = old_mean - young_mean
    
    statistic, p_value = stats.mannwhitneyu(young_data, old_data, alternative='two-sided')
    is_sig = "YES" if p_value < 0.05 else "NO"
    
    print(f"{feature:<35} {young_mean:>12.3f} {old_mean:>12.3f} {diff:>12.3f} {p_value:>10.4f} {is_sig:>5}")
    
    age_results.append({
        'Feature': feature,
        'Young Mean': young_mean,
        'Old Mean': old_mean,
        'Difference (Old-Young)': diff,
        'p-value': p_value,
        'Significant': is_sig
    })

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n\n")
print("="*100)
print("SUMMARY STATISTICS")
print("="*100)
print()

print("Overall narrative statistics:")
print(f"  Mean characters: {results_df['num_characters'].mean():.1f}")
print(f"  Mean words: {results_df['num_words'].mean():.1f}")
print(f"  Mean sentences: {results_df['num_sentences'].mean():.1f}")
print(f"  Mean sentence length: {results_df['avg_sentence_length'].mean():.2f} words/sentence")
print(f"  Mean type-token ratio: {results_df['type_token_ratio'].mean():.3f}")
print(f"  Mean Flesch-Kincaid grade: {results_df['flesch_kincaid_grade'].mean():.2f}")
print(f"  Mean prop. sentences with causal: {results_df['prop_sentences_with_causal'].mean():.3f}")
print()

# ============================================================================
# EXCEL OUTPUT
# ============================================================================
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    # Raw data
    results_df.to_excel(writer, sheet_name='All Narratives', index=False)
    
    # Sex comparison
    sex_df = pd.DataFrame(sex_results)
    sex_df.to_excel(writer, sheet_name='Sex Comparison', index=False)
    
    # Age comparison
    age_df = pd.DataFrame(age_results)
    age_df.to_excel(writer, sheet_name='Age Comparison', index=False)
    
    # Summary statistics by group
    summary_data = []
    for sex in ['male', 'female']:
        for feature in sex_features:
            values = results_df[results_df['sex'] == sex][feature]
            summary_data.append({
                'Group': f'Sex: {sex}',
                'Feature': feature,
                'Mean': values.mean(),
                'Median': values.median(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max()
            })
    
    for age in ['young', 'old']:
        for feature in sex_features:
            values = results_df[results_df['age_group'] == age][feature]
            summary_data.append({
                'Group': f'Age: {age}',
                'Feature': feature,
                'Mean': values.mean(),
                'Median': values.median(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max()
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary Stats', index=False)

print(f"Saved to: {OUTPUT_PATH}")
