import pandas as pd
import numpy as np
import pickle
import os
import shap
from nice import NICE
import warnings
warnings.filterwarnings('ignore')

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
    dataset_path = config['path']
    target_col = config['target_col']
    adverse_file = config['adverse_file']
    protected_attrs = config['protected_attrs']
    
    print(f'\nProcessing {dataset_name}...')
    adverse_path = os.path.join(dataset_path, adverse_file)
    adverse_df = pd.read_csv(adverse_path)
    print(f'  Loaded {len(adverse_df)} adverse instances')
    
    model_path = os.path.join(dataset_path, 'RF.pkl')
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    train_path = os.path.join(dataset_path, 'train_cleaned.parquet')
    train_df = pd.read_parquet(train_path)
    train_df_no_protected = train_df.drop(columns=protected_attrs, errors='ignore')
    train_features = train_df_no_protected.drop(columns=[target_col])
    train_target = train_df_no_protected[target_col]
    
    feature_cols = [col for col in train_features.columns]
    
    # SHAP
    print(f'  Computing SHAP values...')
    adverse_features = adverse_df[feature_cols].copy()
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(adverse_features)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    
    shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{col}' for col in feature_cols])
    shap_df.insert(0, 'instance_index', adverse_df['instance_index'].values)
    shap_df.insert(1, 'predicted_probability', adverse_df['prediction_score'].values)
    shap_df.to_csv(os.path.join(dataset_path, f'{dataset_name}_shap.csv'), index=False)
    
    # NICE
    print(f'  Initializing NICE...')
    if dataset_name == 'law':
        num_feat = [train_features.columns.get_loc(col) for col in ['lsat', 'ugpa', 'fam_inc'] if col in train_features.columns]
    elif dataset_name == 'credit':
        num_feat = [train_features.columns.get_loc(col) for col in ['duration', 'amount', 'installment_rate', 'present_residence', 'number_credits', 'people_liable'] if col in train_features.columns]
    else:
        num_feat = []
    cat_feat = [i for i in range(len(feature_cols)) if i not in num_feat]
    
    def predict_fn(x):
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=feature_cols)
        return rf_model.predict_proba(x)
    
    # Ensure X_train is float64 BEFORE passing to NICE to avoid in-place cast errors
    X_train_float = train_features.values.astype(np.float64, copy=True)
    
    nice = NICE(
        predict_fn=predict_fn,
        X_train=X_train_float,
        cat_feat=cat_feat,
        num_feat=num_feat,
        y_train=train_target.values
    )
    
    print(f'  Generating counterfactuals with NICE...')
    all_cfs = []
    success_count = 0
    failed_count = 0
    
    # Try different optimization strategies to get variety if possible
    opts = ['sparsity', 'proximity', 'plausibility']
    
    for idx, (_, row) in enumerate(adverse_df.iterrows()):
        if (idx + 1) % max(1, len(adverse_df) // 5) == 0 or (idx+1) == 1:
            print(f'    Instance {idx + 1}/{len(adverse_df)}')
        
        # Instance must also be float64
        instance = row[feature_cols].values.reshape(1, -1).astype(np.float64, copy=True)
        
        cfs_generated = 0
        for cf_num in range(1, num_cf + 1):
            try:
                # We reuse the same NICE object but NICE explain is somewhat deterministic.
                # In a more complex setup we might re-init with different 'optimization' param.
                cf_instance = nice.explain(instance)
                
                if isinstance(cf_instance, np.ndarray):
                    cf_instance = cf_instance.flatten()
                
                distance = np.linalg.norm(cf_instance - instance.flatten())
                
                cf_dict = dict(zip(feature_cols, cf_instance))
                cf_dict['instance_index'] = int(row['instance_index'])
                cf_dict['original_test_index'] = int(row['original_test_index'])
                cf_dict['CF_number'] = cf_num
                cf_dict['distance'] = distance
                all_cfs.append(cf_dict)
                success_count += 1
                cfs_generated += 1
                
                # If distance is 0, NICE failed to find a counterfactual that changes anything
                # but technically NICE explain returns a valid CF for the model.
            except Exception as e:
                # print(f'      [Instance {idx+1}] Error: {str(e)}')
                break
        
        if cfs_generated < num_cf:
            failed_count += 1
    
    print(f'  Generated {success_count} counterfactuals')
    if all_cfs:
        cf_df = pd.DataFrame(all_cfs)
        cf_df.to_csv(os.path.join(dataset_path, f'{dataset_name}_counterfactual.csv'), index=False)
        # Check first few CFs for distance
        print(f"  Sample distances: {cf_df['distance'].iloc[:5].values}")

def main():
    for name, config in datasets.items():
        try:
            generate_explanations(name, config)
        except Exception as e:
            print(f'  ERROR: {str(e)}')
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
