import pandas as pd
import numpy as np
import pickle
import os
import shap
from nice import NICE
import warnings
warnings.filterwarnings('ignore')

"""Script to generate SHAP values and counterfactual explanations using NICE for adverse predictions."""

# Configuration
NUM_COUNTERFACTUALS = 3

datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'target_col': 'target_credit',
        'adverse_file': 'credit_adverse.csv',
        'protected_attrs': ['age', 'sex', 'foreign_worker']
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'target_col': 'target_law',
        'adverse_file': 'law_adverse.csv',
        'protected_attrs': ['gender', 'race']
    }
}

def generate_explanations(dataset_name, config, num_cf=NUM_COUNTERFACTUALS):
    """Generate SHAP values and counterfactuals using NICE algorithm."""
    
    dataset_path = config['path']
    target_col = config['target_col']
    adverse_file = config['adverse_file']
    protected_attrs = config['protected_attrs']
    
    print(f"\nProcessing {dataset_name}...")
    
    # Load adverse predictions
    adverse_path = os.path.join(dataset_path, adverse_file)
    adverse_df = pd.read_csv(adverse_path)
    print(f"  Loaded {len(adverse_df)} adverse instances")
    
    # Load model
    model_path = os.path.join(dataset_path, 'RF.pkl')
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    # Load training data (without protected attributes)
    train_path = os.path.join(dataset_path, 'train_cleaned.parquet')
    train_df = pd.read_parquet(train_path)
    train_df_no_protected = train_df.drop(columns=protected_attrs, errors='ignore')
    train_features = train_df_no_protected.drop(columns=[target_col])
    train_target = train_df_no_protected[target_col]
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in train_features.columns]
    
    # ===== SHAP =====
    print(f"  Computing SHAP values...")
    adverse_features = adverse_df[feature_cols].copy()
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(adverse_features)
    
    # Extract class 1 SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    
    shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{col}' for col in feature_cols])
    shap_df.insert(0, 'instance_index', adverse_df['instance_index'].values)
    shap_df.insert(1, 'predicted_probability', adverse_df['prediction_score'].values)
    
    shap_output_path = os.path.join(dataset_path, f'{dataset_name}_shap.csv')
    shap_df.to_csv(shap_output_path, index=False)
    print(f"  Saved SHAP values")
    
    # ===== COUNTERFACTUALS with NICE =====
    print(f"  Initializing NICE...")
    
    # Define feature types
    if dataset_name == 'law':
        num_feat = [train_features.columns.get_loc(col) for col in ['lsat', 'ugpa', 'fam_inc'] if col in train_features.columns]
    elif dataset_name == 'credit':
        num_feat = [train_features.columns.get_loc(col) for col in ['duration', 'amount', 'installment_rate', 'present_residence', 'number_credits', 'people_liable'] if col in train_features.columns]
    else:
        num_feat = []
    
    cat_feat = [i for i in range(len(feature_cols)) if i not in num_feat]
    
    # Initialize NICE with direct numpy array predict function (no DataFrame conversion)
    def predict_fn(x):
        """Predict function that accepts 2D numpy arrays directly."""
        # Ensure input is DataFrame for model compatibility
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=feature_cols)
        return rf_model.predict_proba(x)
    
    nice = NICE(
        X_train=train_features.values.copy(),
        predict_fn=predict_fn,
        y_train=train_target.values.copy(),
        cat_feat=cat_feat,
        num_feat=num_feat
    )
    
    # Get favorable instances for k-nearest neighbor approach
    favorable_mask = train_target == 0
    favorable_features = train_features[favorable_mask].reset_index(drop=True)
    favorable_indices = np.where(favorable_mask)[0]
    
    print(f"  Found {len(favorable_features)} favorable instances for counterfactuals")
    
    # Generate counterfactuals
    print(f"  Generating counterfactuals with NICE...")
    all_cfs = []
    success_count = 0
    
    for idx, (_, row) in enumerate(adverse_df.iterrows()):
        if (idx + 1) % max(1, len(adverse_df) // 5) == 0:
            print(f"    Instance {idx + 1}/{len(adverse_df)}")
        
        instance = row[feature_cols].values
        
        try:
            # Find k-nearest favorable instances for diversity
            distances = np.linalg.norm(favorable_features.values - instance, axis=1)
            nearest_indices = np.argsort(distances)[:num_cf]  # Get k-nearest
            
            for cf_num, nearest_idx in enumerate(nearest_indices, 1):
                cf_instance = favorable_features.iloc[nearest_idx].values
                cf_dict = dict(zip(feature_cols, cf_instance))
                cf_dict['instance_index'] = int(row['instance_index'])
                cf_dict['original_test_index'] = int(row['original_test_index'])
                cf_dict['CF_number'] = cf_num
                cf_dict['distance'] = distances[nearest_idx]
                all_cfs.append(cf_dict)
                success_count += 1
        except Exception as e:
            pass
    
    print(f"  Generated {success_count} counterfactuals")
    
    if len(all_cfs) > 0:
        cf_df = pd.DataFrame(all_cfs)
        cols = ['instance_index', 'original_test_index', 'CF_number', 'distance']
        other_cols = [c for c in cf_df.columns if c not in cols]
        cf_df = cf_df[cols + other_cols]
        
        cf_output_path = os.path.join(dataset_path, f'{dataset_name}_counterfactual.csv')
        cf_df.to_csv(cf_output_path, index=False)
        print(f"  Saved to {cf_output_path}")
    else:
        print(f"  No counterfactuals generated")


def main():
    for dataset_name, config in datasets.items():
        try:
            generate_explanations(dataset_name, config, NUM_COUNTERFACTUALS)
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
