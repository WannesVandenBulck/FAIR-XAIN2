#!/usr/bin/env python
"""
Generate per-narrative metrics CSV with detailed faithfulness and feature coverage metrics.

For each narrative (dataset × provider × condition × instance), loads:
- Extraction JSON (top features extracted by LLM)
- Ground truth JSON (true SHAP features)
- Demographic data (PA values from adverse CSV)
- Predicted probability

Computes detailed metrics:
FAITHFULNESS:
1. Predicted probability accuracy (extracted vs GT)
2. Sign accuracy per rank (1,2,3) + total (checking feature by name, not rank)
3. Rank accuracy per rank (1,2,3) + total
4. PA value accuracy per PA feature
5. SHAP feature value accuracy per rank (1,2,3)
6. Other feature value accuracy per feature
7. Total value accuracy (% of mentioned values that were correct)

FEATURE COVERAGE:
1. SHAP features mentioned (any, per rank 1-3, value mentioned per rank)
2. PA features mentioned (per PA attribute, separately for value)
3. Other features mentioned (per feature, separately for value)
4. Predicted probability mentioned

Output: per_narrative_metrics_<dataset>.csv per dataset
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from glob import glob
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = {
    "credit": {
        "num_instances": 97,
        "protected_attrs": ["age", "sex", "foreign_worker"],
        "adverse_csv": "datasets_prep/data/credit_dataset/credit_adverse.csv",
    },
    "law": {
        "num_instances": 308,
        "protected_attrs": ["gender", "race"],
        "adverse_csv": "datasets_prep/data/law_dataset/law_adverse.csv",
    },
    "saudi": {
        "num_instances": 106,
        "protected_attrs": ["Gender", "Age", "Health_Issues"],
        "adverse_csv": "datasets_prep/data/saudi_dataset/saudi_adverse.csv",
    },
    "student": {
        "num_instances": 73,
        "protected_attrs": ["sex", "age", "health"],
        "adverse_csv": "datasets_prep/data/student_dataset/student_adverse.csv",
    },
}

PROVIDERS = ["grok", "openai", "deepseek"]
CONDITIONS = ["include_pa", "exclude_pa"]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_extraction(dataset, provider, condition, instance_idx):
    """Load extracted features from narrative."""
    pattern = f"results/extractions/{dataset}/{condition}/{provider}/**/instance_{instance_idx}.json"
    matches = glob(pattern, recursive=True)
    if not matches:
        return None
    
    with open(matches[0]) as f:
        return json.load(f)

def load_ground_truth(dataset, instance_idx):
    """Load ground truth SHAP features."""
    path = f"results/ground_truth/json/{dataset}/instance_{instance_idx}.json"
    if not Path(path).exists():
        return None
    
    with open(path) as f:
        return json.load(f)

def load_adverse_csv(dataset):
    """Load demographic data with PA values."""
    csv_path = DATASETS[dataset]["adverse_csv"]
    return pd.read_csv(csv_path)

# ============================================================================
# GROUND TRUTH PARSING
# ============================================================================

def parse_ground_truth_features(gt_dict):
    """
    Extract features from ground truth.
    
    Returns: {
        'shap_features': [
            {'rank': 1, 'name': 'feature_x', 'sign': 1, 'value': 0.5},
            ...
        ],
        'predicted_probability': 0.72,
    }
    """
    result = {
        'shap_features': [],
        'predicted_probability': None,
    }
    
    # Extract top 3 SHAP features from most_important_features array
    if "most_important_features" in gt_dict:
        features = gt_dict["most_important_features"]
        if isinstance(features, list):
            for feature in features:
                if len(result['shap_features']) < 3:
                    f = {
                        'rank': feature.get('rank'),
                        'name': feature.get('name'),
                        'sign': feature.get('sign'),
                        'value': feature.get('value'),
                    }
                    result['shap_features'].append(f)
    
    # Extract predicted probability
    if "predicted_probability" in gt_dict:
        result['predicted_probability'] = gt_dict["predicted_probability"]
    
    return result

def parse_extraction_features(extraction_dict):
    """
    Extract features from LLM extraction.
    
    Returns: {
        'shap_features': [
            {'rank': 1, 'name': 'feature_x', 'sign': 1, 'value': 0.5},
            ...
        ],
        'predicted_probability': 0.72,
    }
    """
    result = {
        'shap_features': [],
        'predicted_probability': None,
    }
    
    # Extract from most_important_features
    if "most_important_features" in extraction_dict:
        features = extraction_dict["most_important_features"]
        if isinstance(features, list):
            for feature in features[:3]:
                f = {
                    'rank': feature.get('rank'),
                    'name': feature.get("name"),
                    'sign': feature.get("sign"),
                    'value': feature.get("value"),
                }
                result['shap_features'].append(f)
    
    # Extract predicted probability
    if "predicted_probability" in extraction_dict:
        result['predicted_probability'] = extraction_dict["predicted_probability"]
    
    return result

# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def compute_sign_accuracy(extracted_features, gt_features, dataset, pa_attrs):
    """
    Compute sign accuracy per rank (1,2,3) and total.
    
    For each extracted rank, get the feature name, find that feature in GT,
    and check if the sign matches.
    
    Returns: {'sign_rank_1_accuracy': 1/0/NaN, ..., 'sign_total_accuracy': 0-3}
    """
    result = {}
    total_correct = 0
    
    # Create GT feature lookup by name
    gt_by_name = {f['name']: f for f in gt_features}
    
    for rank in [1, 2, 3]:
        result[f'sign_rank_{rank}_accuracy'] = np.nan
        
        # Find extracted feature at this rank
        ext_feat = next((f for f in extracted_features if f['rank'] == rank), None)
        if not ext_feat or ext_feat['name'] is None or ext_feat['sign'] is None:
            continue
        
        # Find same feature in GT by name
        gt_feat = gt_by_name.get(ext_feat['name'])
        if not gt_feat or gt_feat['sign'] is None:
            result[f'sign_rank_{rank}_accuracy'] = np.nan
            continue
        
        # Compare signs
        if ext_feat['sign'] == gt_feat['sign']:
            result[f'sign_rank_{rank}_accuracy'] = 1
            total_correct += 1
        else:
            result[f'sign_rank_{rank}_accuracy'] = 0
    
    # Total sign accuracy (count of correct out of 3)
    result['sign_total_accuracy'] = total_correct
    
    return result

def compute_rank_accuracy(extracted_features, gt_features):
    """
    Compute rank accuracy per rank (1,2,3) and total.
    
    For each extracted rank, check if the feature at that rank is actually
    at that rank in the GT.
    
    Returns: {'rank_1_accuracy': 1/0/NaN, ..., 'rank_total_accuracy': 0-3}
    """
    result = {}
    total_correct = 0
    
    # Create GT feature lookup by name
    gt_by_name = {f['name']: f for f in gt_features}
    
    for rank in [1, 2, 3]:
        result[f'rank_{rank}_accuracy'] = np.nan
        
        # Find extracted feature at this rank
        ext_feat = next((f for f in extracted_features if f['rank'] == rank), None)
        if not ext_feat or ext_feat['name'] is None:
            continue
        
        # Find same feature in GT by name
        gt_feat = gt_by_name.get(ext_feat['name'])
        if not gt_feat:
            result[f'rank_{rank}_accuracy'] = 0
            continue
        
        # Check if ranks match
        if ext_feat['rank'] == gt_feat['rank']:
            result[f'rank_{rank}_accuracy'] = 1
            total_correct += 1
        else:
            result[f'rank_{rank}_accuracy'] = 0
    
    # Total rank accuracy
    result['rank_total_accuracy'] = total_correct
    
    return result

def compute_predicted_probability_accuracy(extracted_pp, gt_pp):
    """
    Compare extracted predicted probability to ground truth.
    Uses tolerance for floating-point comparison to handle rounding differences.
    """
    if extracted_pp is None or gt_pp is None:
        return np.nan
    
    # Convert to float in case they're strings
    try:
        ext_pp = float(extracted_pp)
        gt_pp_float = float(gt_pp)
    except (ValueError, TypeError):
        return np.nan
    
    # Use tolerance for floating-point comparison (handles rounding like 0.992 vs 0.99)
    if np.isclose(ext_pp, gt_pp_float, atol=0.01):
        return 1
    else:
        return 0

def compute_pa_value_accuracy(dataset, extraction_dict, adverse_df, instance_idx):
    """
    Compute value accuracy for each PA feature by comparing extracted values
    against demographic data in adverse CSV.
    
    Returns: {'pa_<attr>_value_accuracy': 1/0/NaN for each PA}
    """
    result = {}
    pa_attrs = DATASETS[dataset]["protected_attrs"]
    
    # Get true PA values from adverse CSV
    true_pa_values = {}
    if instance_idx in adverse_df.index:
        for pa in pa_attrs:
            try:
                true_pa_values[pa] = adverse_df.loc[instance_idx, pa]
            except (KeyError, IndexError):
                pass
    
    # Get extracted PA values from extraction's features array
    extracted_pa_values = {}
    if "features" in extraction_dict:
        features = extraction_dict["features"]
        if isinstance(features, list):
            for feat in features:
                name = feat.get('name')
                value = feat.get('value')
                if name:
                    # Case-insensitive match for PA names
                    for pa in pa_attrs:
                        if name.lower() == pa.lower():
                            # Check if feature was mentioned and has a value
                            if feat.get('mentioned') == 1 and value is not None and value != "NaN":
                                extracted_pa_values[pa] = value
                            break
    
    # Compute accuracy for each PA
    for pa in pa_attrs:
        result[f'pa_{pa}_value_accuracy'] = np.nan
        
        # Both values must exist to compare
        if pa not in true_pa_values or pa not in extracted_pa_values:
            continue
        
        # Compare values
        true_val = true_pa_values[pa]
        ext_val = extracted_pa_values[pa]
        
        # Handle type conversions for comparison
        try:
            if isinstance(true_val, (int, float)) and isinstance(ext_val, (int, float)):
                result[f'pa_{pa}_value_accuracy'] = 1 if true_val == ext_val else 0
            else:
                # String comparison after converting both to string
                result[f'pa_{pa}_value_accuracy'] = 1 if str(true_val) == str(ext_val) else 0
        except (ValueError, TypeError):
            result[f'pa_{pa}_value_accuracy'] = 0
    
    return result

def compute_shap_feature_value_accuracy(extracted_features, gt_features):
    """
    Compute value accuracy for each of top 3 SHAP features.
    
    Returns: {'shap_feature_1_value_accuracy': 1/0/NaN, ...}
    """
    result = {}
    
    # Create GT feature lookup by name
    gt_by_name = {f['name']: f for f in gt_features}
    
    for rank in [1, 2, 3]:
        result[f'shap_feature_{rank}_value_accuracy'] = np.nan
        
        # Find extracted feature at this rank
        ext_feat = next((f for f in extracted_features if f['rank'] == rank), None)
        if not ext_feat or ext_feat['name'] is None or ext_feat['value'] is None:
            continue
        
        # Find same feature in GT by name
        gt_feat = gt_by_name.get(ext_feat['name'])
        if not gt_feat or gt_feat['value'] is None:
            result[f'shap_feature_{rank}_value_accuracy'] = np.nan
            continue
        
        # Compare values
        if ext_feat['value'] == gt_feat['value']:
            result[f'shap_feature_{rank}_value_accuracy'] = 1
        else:
            result[f'shap_feature_{rank}_value_accuracy'] = 0
    
    return result

def compute_other_feature_value_accuracy(shap_features, extraction_dict, gt_dict, all_other_features):
    """
    Compute value accuracy for other features (not PA, not top 3 SHAP).
    
    Returns: {'other_<feature>_value_accuracy': 1/0/NaN for each feature}
    """
    result = {}
    
    # Get top 3 SHAP feature names
    shap_names = {f['name'] for f in shap_features if f['name'] is not None}
    
    # Create GT feature lookup from features array
    gt_by_name = {}
    if "features" in gt_dict:
        features = gt_dict["features"]
        if isinstance(features, list):
            for feat in features:
                name = feat.get('name')
                if name and name not in shap_names:  # Exclude top 3 SHAP
                    gt_by_name[name] = feat
    
    # Create extracted feature lookup from features array (excludes top 3 SHAP)
    ext_by_name = {}
    if "features" in extraction_dict:
        features = extraction_dict["features"]
        if isinstance(features, list):
            for feat in features:
                name = feat.get('name')
                if name and name not in shap_names:  # Exclude top 3 SHAP
                    ext_by_name[name] = feat
    
    # Compute accuracy for each other feature
    for feature in all_other_features:
        result[f'other_{feature}_value_accuracy'] = np.nan
        
        # Check if feature was extracted
        ext_feat = ext_by_name.get(feature)
        if not ext_feat:
            continue
        
        ext_value = ext_feat.get('value')
        if ext_value is None or ext_value == "NaN":
            continue
        
        # Check if feature exists in GT
        gt_feat = gt_by_name.get(feature)
        if not gt_feat:
            result[f'other_{feature}_value_accuracy'] = np.nan
            continue
        
        gt_value = gt_feat.get('value')
        if gt_value is None or gt_value == "NaN":
            result[f'other_{feature}_value_accuracy'] = np.nan
            continue
        
        # Compare values
        if ext_value == gt_value:
            result[f'other_{feature}_value_accuracy'] = 1
        else:
            result[f'other_{feature}_value_accuracy'] = 0
    
    return result

def compute_total_value_accuracy(sign_acc, rank_acc, pp_acc, pa_acc, shap_acc, other_acc):
    """
    Total value accuracy: % of all mentioned values that were correct.
    
    Count all value accuracies that are not NaN, sum the 1s, divide by total count.
    """
    all_accuracies = []
    
    # Predicted probability accuracy
    if not np.isnan(pp_acc):
        all_accuracies.append(pp_acc)
    
    # PA value accuracies
    for key, val in pa_acc.items():
        if not np.isnan(val):
            all_accuracies.append(val)
    
    # SHAP value accuracies
    for key, val in shap_acc.items():
        if not np.isnan(val):
            all_accuracies.append(val)
    
    # Other feature value accuracies
    for key, val in other_acc.items():
        if not np.isnan(val):
            all_accuracies.append(val)
    
    if not all_accuracies:
        return np.nan
    
    return sum(all_accuracies) / len(all_accuracies)

def compute_feature_coverage(extracted_features, extracted_dict, gt_dict, dataset, pa_attrs, all_other_features):
    """
    Compute feature coverage metrics.
    """
    result = {}
    
    # Load features array for mentioned checks
    features_by_name = {}
    if "features" in extracted_dict:
        features = extracted_dict["features"]
        if isinstance(features, list):
            for feat in features:
                name = feat.get('name')
                if name:
                    features_by_name[name] = feat
    
    # SHAP features mentioned (from extracted_features list - top 3)
    # Check mentioned field from features array for consistency
    shap_mentioned_count = 0
    for rank in [1, 2, 3]:
        ext_feat = next((f for f in extracted_features if f['rank'] == rank), None)
        
        # Check if feature is mentioned in features array
        mentioned = 0
        if ext_feat and ext_feat['name'] is not None:
            feat_info = features_by_name.get(ext_feat['name'])
            mentioned = 1 if (feat_info and feat_info.get('mentioned') == 1) else 0
        
        result[f'shap_feature_{rank}_mentioned'] = mentioned
        shap_mentioned_count += mentioned
        
        # Value mentioned: check if value exists and is not "NaN"
        value_mentioned = 0
        if ext_feat and ext_feat['value'] is not None and ext_feat['value'] != "NaN":
            value_mentioned = 1
        result[f'shap_feature_{rank}_value_mentioned'] = value_mentioned
    
    result['shap_features_mentioned'] = 1 if shap_mentioned_count > 0 else 0
    
    # PA features mentioned (from features array)
    for pa in pa_attrs:
        feat = features_by_name.get(pa)
        pa_mentioned = 1 if (feat and feat.get('mentioned') == 1) else 0
        result[f'pa_{pa}_mentioned'] = pa_mentioned
        
        pa_value_mentioned = 0
        if feat and feat.get('mentioned') == 1:
            value = feat.get('value')
            pa_value_mentioned = 1 if (value is not None and value != "NaN") else 0
        result[f'pa_{pa}_value_mentioned'] = pa_value_mentioned
    
    # Other features mentioned (from features array, excluding top 3 SHAP)
    shap_names = {f['name'] for f in extracted_features if f['name'] is not None}
    
    for feature in all_other_features:
        feat = features_by_name.get(feature)
        mentioned = 1 if (feat and feat.get('mentioned') == 1) else 0
        result[f'other_{feature}_mentioned'] = mentioned
        
        value_mentioned = 0
        if feat and feat.get('mentioned') == 1:
            value = feat.get('value')
            value_mentioned = 1 if (value is not None and value != "NaN") else 0
        result[f'other_{feature}_value_mentioned'] = value_mentioned
    
    # Predicted probability mentioned
    result['predicted_probability_mentioned'] = 1 if extracted_dict.get('predicted_probability') is not None else 0
    
    return result

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def identify_all_features(dataset):
    """
    Identify all possible other features across all instances in a dataset.
    Excludes protected attributes (PAs) which are handled separately.
    """
    all_other_features = set()
    pa_attrs = set(attr.lower() for attr in DATASETS[dataset]["protected_attrs"])
    
    for instance_idx in range(DATASETS[dataset]["num_instances"]):
        gt_path = f"results/ground_truth/json/{dataset}/instance_{instance_idx}.json"
        if Path(gt_path).exists():
            with open(gt_path) as f:
                gt = json.load(f)
                # Get features from the "features" array (excludes top 3 SHAP features)
                if "features" in gt:
                    features = gt["features"]
                    if isinstance(features, list):
                        for feat in features:
                            name = feat.get('name')
                            if name and name.lower() not in pa_attrs:  # Exclude PAs
                                all_other_features.add(name)
    
    return sorted(list(all_other_features))

def process_dataset(dataset):
    """Process all narratives for a single dataset."""
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset}")
    print(f"{'='*80}")
    
    num_instances = DATASETS[dataset]["num_instances"]
    pa_attrs = DATASETS[dataset]["protected_attrs"]
    adverse_df = load_adverse_csv(dataset).set_index('instance_index')
    
    # Identify all other features
    all_other_features = identify_all_features(dataset)
    print(f"Identified {len(all_other_features)} other features: {all_other_features}")
    
    records = []
    
    for instance_idx in range(num_instances):
        for provider in PROVIDERS:
            for condition in CONDITIONS:
                # Load data
                extraction = load_extraction(dataset, provider, condition, instance_idx)
                gt_json = load_ground_truth(dataset, instance_idx)
                
                if not extraction or not gt_json:
                    continue
                
                # Parse
                gt_features = parse_ground_truth_features(gt_json)
                ext_features = parse_extraction_features(extraction)
                
                # Get PA values from adverse CSV
                pa_values = {}
                if instance_idx in adverse_df.index:
                    for pa in pa_attrs:
                        pa_values[f'pa_{pa}'] = adverse_df.loc[instance_idx, pa]
                
                # Compute metrics
                sign_acc = compute_sign_accuracy(ext_features['shap_features'], gt_features['shap_features'], dataset, pa_attrs)
                rank_acc = compute_rank_accuracy(ext_features['shap_features'], gt_features['shap_features'])
                pp_acc = compute_predicted_probability_accuracy(ext_features['predicted_probability'], gt_features['predicted_probability'])
                pa_acc = compute_pa_value_accuracy(dataset, extraction, adverse_df, instance_idx)
                shap_acc = compute_shap_feature_value_accuracy(ext_features['shap_features'], gt_features['shap_features'])
                other_acc = compute_other_feature_value_accuracy(ext_features['shap_features'], extraction, gt_json, all_other_features)
                total_val_acc = compute_total_value_accuracy(sign_acc, rank_acc, pp_acc, pa_acc, shap_acc, other_acc)
                
                coverage = compute_feature_coverage(ext_features['shap_features'], extraction, gt_json, dataset, pa_attrs, all_other_features)
                
                # Build record
                record = {
                    'dataset': dataset,
                    'provider': provider,
                    'condition': condition,
                    'instance_id': instance_idx,
                    'predicted_probability': gt_features['predicted_probability'],
                }
                
                # Add PA values
                record.update(pa_values)
                
                # Add faithfulness metrics
                record.update(sign_acc)
                record.update(rank_acc)
                record['predicted_probability_accuracy'] = pp_acc
                record.update(pa_acc)
                record.update(shap_acc)
                record.update(other_acc)
                record['total_value_accuracy'] = total_val_acc
                
                # Add feature coverage metrics
                record.update(coverage)
                
                records.append(record)
    
    # Create DataFrame and export
    df = pd.DataFrame(records)
    
    # Reorder columns logically: 
    # 1. Metadata (dataset, provider, condition, instance_id, predicted_probability, pa values)
    # 2. Faithfulness metrics (sign/rank accuracies, pp accuracy)
    # 3. For each feature category: mentioned → value_mentioned → value_accuracy
    # 4. Total value accuracy
    
    # Start with metadata and faithfulness columns
    meta_cols = ['dataset', 'provider', 'condition', 'instance_id', 'predicted_probability']
    
    # PA value columns from adverse CSV
    pa_cols = [col for col in df.columns if col.startswith('pa_') and '_' not in col.split('_')[1]]
    pa_cols = sorted([col for col in df.columns if col.startswith('pa_') and not any(x in col for x in ['mentioned', 'accuracy'])])
    
    # Faithfulness metrics
    faithfulness_cols = [
        'sign_rank_1_accuracy', 'sign_rank_2_accuracy', 'sign_rank_3_accuracy', 'sign_total_accuracy',
        'rank_1_accuracy', 'rank_2_accuracy', 'rank_3_accuracy', 'rank_total_accuracy',
        'predicted_probability_accuracy'
    ]
    faithfulness_cols = [col for col in faithfulness_cols if col in df.columns]
    
    # Feature coverage and accuracy columns, organized by feature and metric type
    remaining_cols = [col for col in df.columns if col not in meta_cols + pa_cols + faithfulness_cols and col != 'total_value_accuracy']
    
    # Organize by feature: mentioned → value_mentioned → value_accuracy
    organized_cols = []
    
    # Extract unique feature groups (shap_feature_*, pa_*, other_*)
    feature_groups = set()
    for col in remaining_cols:
        if '_mentioned' in col or '_value_mentioned' in col or '_value_accuracy' in col:
            # Extract base feature name
            parts = col.split('_')
            if col.startswith('shap_feature_'):
                # shap_feature_1_mentioned or shap_feature_1_value_mentioned or shap_feature_1_value_accuracy
                rank = parts[2]  # Get "1", "2", or "3"
                metric_type = 'mentioned' if '_mentioned' in col and '_value' not in col else ('value_mentioned' if '_value_mentioned' in col else 'value_accuracy')
                if metric_type == 'value_accuracy':
                    metric_type = 'value_accuracy'
                feature_groups.add(('shap_feature', rank, metric_type))
            elif col.startswith('pa_') and col not in pa_cols:
                # pa_age_mentioned or pa_age_value_mentioned or pa_age_value_accuracy
                base_name = col.replace('_mentioned', '').replace('_value_mentioned', '').replace('_value_accuracy', '')
                if base_name.startswith('pa_'):
                    pa_name = base_name[3:]  # Remove "pa_" prefix
                    metric_type = 'mentioned' if '_mentioned' in col and '_value' not in col else ('value_mentioned' if '_value_mentioned' in col else 'value_accuracy')
                    feature_groups.add(('pa', pa_name, metric_type))
            elif col.startswith('other_'):
                # other_<name>_mentioned or other_<name>_value_mentioned or other_<name>_value_accuracy
                base_name = col.replace('_mentioned', '').replace('_value_mentioned', '').replace('_value_accuracy', '')
                if base_name.startswith('other_'):
                    other_name = base_name[6:]  # Remove "other_" prefix
                    metric_type = 'mentioned' if '_mentioned' in col and '_value' not in col else ('value_mentioned' if '_value_mentioned' in col else 'value_accuracy')
                    feature_groups.add(('other', other_name, metric_type))
    
    # Build organized column list by iterating through features
    # Group 1: shap_features_mentioned
    if 'shap_features_mentioned' in remaining_cols:
        organized_cols.append('shap_features_mentioned')
    
    # shap_feature_X (1,2,3): mentioned → value_mentioned → value_accuracy
    for rank in ['1', '2', '3']:
        # mentioned
        col = f'shap_feature_{rank}_mentioned'
        if col in remaining_cols:
            organized_cols.append(col)
    
    for rank in ['1', '2', '3']:
        # value_mentioned
        col = f'shap_feature_{rank}_value_mentioned'
        if col in remaining_cols:
            organized_cols.append(col)
    
    for rank in ['1', '2', '3']:
        # value_accuracy
        col = f'shap_feature_{rank}_value_accuracy'
        if col in remaining_cols:
            organized_cols.append(col)
    
    # PA features: mentioned → value_mentioned → value_accuracy
    pa_feature_names = set()
    for col in remaining_cols:
        if col.startswith('pa_'):
            # Extract PA name - check longer patterns first to avoid substring matches
            if '_mentioned' in col or '_value_mentioned' in col or '_value_accuracy' in col:
                for metric in ['_value_accuracy', '_value_mentioned', '_mentioned']:
                    if metric in col:
                        pa_name = col.replace('pa_', '').replace(metric, '')
                        pa_feature_names.add(pa_name)
                        break
    
    for pa_name in sorted(pa_feature_names):
        # mentioned
        col = f'pa_{pa_name}_mentioned'
        if col in remaining_cols:
            organized_cols.append(col)
    
    for pa_name in sorted(pa_feature_names):
        # value_mentioned
        col = f'pa_{pa_name}_value_mentioned'
        if col in remaining_cols:
            organized_cols.append(col)
    
    for pa_name in sorted(pa_feature_names):
        # value_accuracy
        col = f'pa_{pa_name}_value_accuracy'
        if col in remaining_cols:
            organized_cols.append(col)
    
    # Other features: mentioned → value_mentioned → value_accuracy
    other_feature_names = set()
    for col in remaining_cols:
        if col.startswith('other_'):
            # Extract feature name - check longer patterns first to avoid substring matches
            for metric in ['_value_accuracy', '_value_mentioned', '_mentioned']:
                if metric in col:
                    other_name = col.replace('other_', '').replace(metric, '')
                    other_feature_names.add(other_name)
                    break
    
    for other_name in sorted(other_feature_names):
        # mentioned
        col = f'other_{other_name}_mentioned'
        if col in remaining_cols:
            organized_cols.append(col)
    
    for other_name in sorted(other_feature_names):
        # value_mentioned
        col = f'other_{other_name}_value_mentioned'
        if col in remaining_cols:
            organized_cols.append(col)
    
    for other_name in sorted(other_feature_names):
        # value_accuracy
        col = f'other_{other_name}_value_accuracy'
        if col in remaining_cols:
            organized_cols.append(col)
    
    # Add predicted_probability_mentioned if it exists
    if 'predicted_probability_mentioned' in remaining_cols:
        organized_cols.append('predicted_probability_mentioned')
    
    # Add total_value_accuracy at the end
    if 'total_value_accuracy' in df.columns:
        organized_cols.append('total_value_accuracy')
    
    # Build final column order
    final_cols = meta_cols + pa_cols + faithfulness_cols + organized_cols
    
    # Ensure all columns are included
    final_cols = [col for col in final_cols if col in df.columns]
    
    # Reorder DataFrame
    df = df[final_cols]
    
    output_path = f"results/fairness_eval/per_narrative_metrics_{dataset}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} records for {dataset}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Output: {output_path}")
    
    return df

def main():
    print("="*80)
    print("GENERATE PER-NARRATIVE METRICS (DETAILED)")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)
    
    for dataset in DATASETS:
        process_dataset(dataset)
    
    print("\n" + "="*80)
    print(f"Completed: {datetime.now().isoformat()}")
    print("="*80)

if __name__ == "__main__":
    main()
