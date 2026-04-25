import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import sys

# Add parent path to import llm_tools modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_tools.prompts.prompt_credit import ATTRIBUTE_VALUE_MAPPINGS as CREDIT_MAPPINGS, map_attribute_value as credit_map_attribute_value
from llm_tools.prompts.prompt_law import ATTRIBUTE_VALUE_MAPPINGS as LAW_MAPPINGS, map_attribute_value as law_map_attribute_value

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def to_python_scalar(value):
    """Convert numpy scalars to native Python scalars for JSON-safe keys/values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value

# Configuration - Feature types from prep files
datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'counterfactual_file': 'credit_counterfactual.csv',
        'adverse_file': 'credit_adverse.csv',
        'test_file': 'test_cleaned.parquet',
        'numerical_features': ['duration', 'amount', 'installment_rate', 'present_residence', 'number_credits', 'people_liable'],
        'categorical_features': ['status', 'credit_history', 'savings', 'employment_duration', 'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker', 'purpose', 'other_debtors', 'property']
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'counterfactual_file': 'law_counterfactual.csv',
        'adverse_file': 'law_adverse.csv',
        'test_file': 'test_cleaned.parquet',
        'numerical_features': ['decile1', 'decile3', 'lsat', 'ugpa'],
        'categorical_features': ['fam_inc', 'fulltime']
        # Note: gender and race1 are protected attributes excluded from counterfactual CSVs
    }
}

def analyze_counterfactuals(dataset_name, config):
    """Analyze counterfactual changes for each instance."""
    
    dataset_path = config['path']
    cf_file = config['counterfactual_file']
    adverse_file = config['adverse_file']
    numerical_features = set(config.get('numerical_features', []))
    categorical_features = set(config.get('categorical_features', []))
    
    # Load dataframes
    cf_path = os.path.join(dataset_path, cf_file)
    adverse_path = os.path.join(dataset_path, adverse_file)
    
    cf_df = pd.read_csv(cf_path)
    adverse_df = pd.read_csv(adverse_path)
    
    print(f"\n{dataset_name.upper()}")
    print(f"  Loaded {len(cf_df)} counterfactuals for {len(adverse_df)} adverse instances")
    
    # Metadata columns to exclude from analysis
    metadata_cols = {'instance_index', 'original_test_index', 'CF_number', 'distance_to_original', 'distance'}

    # Get analyzable feature columns: present in both CF and adverse files, excluding metadata
    feature_cols = [
        col for col in cf_df.columns
        if col not in metadata_cols and col in adverse_df.columns
    ]
    
    num_numerical = len([f for f in feature_cols if f in numerical_features])
    num_categorical = len([f for f in feature_cols if f in categorical_features])
    
    print(f"  Analyzing {len(feature_cols)} features ({num_numerical} numerical, {num_categorical} categorical)")
    
    # Store results for all instances
    results = {}
    
    # Process each instance
    for instance_idx in adverse_df['instance_index'].unique():
        # Get original adverse instance
        adverse_row = adverse_df[adverse_df['instance_index'] == instance_idx].iloc[0]
        
        # Get all counterfactuals for this instance
        cf_instances = cf_df[cf_df['instance_index'] == instance_idx]
        num_cfs = len(cf_instances)
        
        instance_stats = {}
        
        # Analyze each feature
        for feature in feature_cols:
            original_value = adverse_row[feature]
            cf_values = cf_instances[feature].values
            
            # Count how many changed
            changed_mask = cf_values != original_value
            num_changed = changed_mask.sum()
            percentage_changed = (num_changed / num_cfs) * 100 if num_cfs > 0 else 0
            
            feature_stats = {
                'percentage_changed': round(percentage_changed, 2),
                'num_changed': num_changed,
                'num_total': num_cfs,
                'original_value': original_value
            }
            
            if num_changed > 0:
                changed_values = cf_values[changed_mask]
                
                if feature in numerical_features:
                    # Numeric feature analysis
                    changes = changed_values.astype(float) - float(original_value)
                    feature_stats['average_change'] = round(float(np.mean(changes)), 4)
                    feature_stats['min_change'] = round(float(np.min(changes)), 4)
                    feature_stats['max_change'] = round(float(np.max(changes)), 4)
                    feature_stats['std_change'] = round(float(np.std(changes)), 4)
                    feature_stats['changes'] = [round(float(c), 4) for c in changes]
                else:
                    # Categorical feature analysis with mapping to readable names
                    value_counts = Counter(changed_values)
                    
                    # Determine which mapping to use based on dataset
                    if dataset_name == 'credit':
                        map_func = credit_map_attribute_value
                    else:
                        map_func = law_map_attribute_value
                    
                    # Create distribution with mapped (readable) names as keys
                    feature_stats['distribution'] = {}
                    for k, v in value_counts.items():
                        mapped_name = map_func(feature, to_python_scalar(k))
                        feature_stats['distribution'][str(mapped_name)] = int(v)
                    
                    feature_stats['unique_values'] = [to_python_scalar(k) for k in value_counts.keys()]
            
            instance_stats[feature] = feature_stats
        
        results[f"instance_{instance_idx}"] = {
            'num_counterfactuals': num_cfs,
            'features': instance_stats
        }
    
    # Save results to JSON
    output_path = os.path.join(dataset_path, f'{dataset_name}_counterfactual_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"  Saved analysis to {output_path}")
    
    # Print sample statistics
    print(f"  Sample statistics (first instance):")
    first_instance = list(results.values())[0]
    print(f"    Total counterfactuals: {first_instance['num_counterfactuals']}")
    
    # Show a few features
    for feature_name, stats in list(first_instance['features'].items())[:3]:
        print(f"    {feature_name}:")
        print(f"      Changed in {stats['percentage_changed']}% of cases ({stats['num_changed']}/{stats['num_total']})")
        if 'average_change' in stats:
            print(f"      Average change: {stats['average_change']}")
        elif 'distribution' in stats:
            print(f"      Distribution: {stats['distribution']}")
    
    return results

def create_summary_csv(dataset_name, config, results):
    """Create a summary CSV with one row per feature per instance."""
    
    dataset_path = config['path']
    numerical_features = set(config.get('numerical_features', []))
    categorical_features = set(config.get('categorical_features', []))
    
    rows = []
    for instance_key, instance_data in results.items():
        instance_num = int(instance_key.split('_')[1])
        
        for feature_name, feature_stats in instance_data['features'].items():
            row = {
                'instance_index': instance_num,
                'feature_name': feature_name,
                'feature_type': 'numerical' if feature_name in numerical_features else 'categorical',
                'percentage_changed': feature_stats['percentage_changed'],
                'num_changed': int(feature_stats['num_changed']),
                'num_total': int(feature_stats['num_total']),
                'original_value': float(feature_stats['original_value']) if isinstance(feature_stats['original_value'], (int, float, np.integer, np.floating)) else str(feature_stats['original_value'])
            }
            
            # Add numeric-specific columns
            if feature_name in numerical_features:
                row['average_change'] = feature_stats.get('average_change', np.nan)
                row['min_change'] = feature_stats.get('min_change', np.nan)
                row['max_change'] = feature_stats.get('max_change', np.nan)
                row['std_change'] = feature_stats.get('std_change', np.nan)
            else:
                row['distribution'] = json.dumps(feature_stats.get('distribution', {}), cls=NumpyEncoder)
                row['unique_values'] = ','.join(str(v) for v in feature_stats.get('unique_values', []))
            
            rows.append(row)
    
    # Create dataframe
    summary_df = pd.DataFrame(rows)
    
    # Sort by instance and feature
    summary_df = summary_df.sort_values(['instance_index', 'feature_name']).reset_index(drop=True)
    
    # Save to CSV
    output_path = os.path.join(dataset_path, f'{dataset_name}_counterfactual_summary.csv')
    summary_df.to_csv(output_path, index=False)
    
    print(f"  Summary CSV saved to {output_path}")

def main():
    """Main function to analyze all counterfactual files."""
    
    print("=" * 70)
    print("COUNTERFACTUAL ANALYSIS: Feature Change Statistics")
    print("=" * 70)
    
    for dataset_name, config in datasets.items():
        try:
            results = analyze_counterfactuals(dataset_name, config)
            create_summary_csv(dataset_name, config, results)
        except FileNotFoundError as e:
            print(f"  Skipping {dataset_name}: File not found ({str(e)})")
        except Exception as e:
            print(f"  ERROR processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
