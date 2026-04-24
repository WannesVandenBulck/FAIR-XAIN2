import pandas as pd
import numpy as np
import pickle
import os
import shap
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

"""Script to generate SHAP values and counterfactual explanations for adverse predictions."""

# Configuration
# Target encoding across all datasets:
#   1 = adverse/bad class (bad credit, failed exam)
#   0 = favorable/good class (good credit, passed exam)
NUM_COUNTERFACTUALS = 10  # Change this to generate different numbers of counterfactuals

# Define dataset configurations
datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'target_col': 'target_credit',
        'adverse_file': 'credit_adverse.csv',
        'frozen_features': ['age', 'sex', 'foreign_worker']  # Features NOT to be changed in counterfactuals
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'target_col': 'target_law',
        'adverse_file': 'law_adverse.csv',
        'frozen_features': ['gender', 'race']  # Features NOT to be changed in counterfactuals
    }
}

def generate_explanations(dataset_name, config, num_cf=NUM_COUNTERFACTUALS):
    """Generate SHAP values and counterfactuals for adverse predictions.
    
    Only processes instances with class 1 (adverse outcome).
    Adds target variable with standardized name to SHAP and CF outputs.
    
    For law dataset: Uses model trained WITHOUT protected attributes (gender, race).
    For credit dataset: Uses model trained WITHOUT protected attributes (age, sex, foreign_worker).
    
    Frozen features (specified in config) are kept constant across all counterfactuals.
    """
        
    dataset_path = config['path']
    target_col = config['target_col']
    adverse_file = config['adverse_file']
    frozen_features = config.get('frozen_features', [])  # Features to keep constant in counterfactuals
    
    # Load adverse predictions (instance_index is now an explicit column, not index)
    adverse_path = os.path.join(dataset_path, adverse_file)
    adverse_df = pd.read_csv(adverse_path)
        
    # Load the RF model
    if dataset_name == 'law':
        model_path = os.path.join(dataset_path, 'RF.pkl')
        protected_attributes = ['gender', 'race']
    elif dataset_name == 'credit':
        model_path = os.path.join(dataset_path, 'RF.pkl')
        protected_attributes = ['age', 'sex', 'foreign_worker']
    
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    print(f"  Frozen features (constant in counterfactuals): {frozen_features}")
    
    # Load test data for context
    test_path = os.path.join(dataset_path, 'test_cleaned.parquet')
    test_df = pd.read_parquet(test_path)
    test_features = test_df.drop(columns=[target_col])
    
    # Get only the feature columns (exclude metadata like instance_index, predictions, and target)
    all_feature_cols = [col for col in adverse_df.columns if col not in 
                    ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', target_col]]
    
    # Exclude protected attributes from features used by model
    feature_cols = [col for col in all_feature_cols if col not in protected_attributes]
    adverse_features = adverse_df[feature_cols].copy()
    
    # Also remove protected attributes from test features
    test_features = test_features.drop(columns=protected_attributes, errors='ignore')
    
    # ===== SHAP VALUES =====
    # Use TreeSHAP for RF models
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(adverse_features)
    
    # TreeSHAP returns shape (n_instances, n_features, n_classes) for binary classification
    # We need class 1 (bad/adverse class) SHAP values
    if isinstance(shap_values, list):
        # If returned as list of arrays [class_0_shap, class_1_shap]
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        # If returned as 3D array (n_instances, n_features, 2), extract class 1
        shap_values = shap_values[:, :, 1]
    
    # Ensure it's 2D (n_instances, n_features)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)
    
    # Create dataframe with SHAP values (with explicit instance_index column)
    shap_df = pd.DataFrame(
        shap_values,
        columns=[f'SHAP_{col}' for col in feature_cols]
    )
    
    # Add instance_index and original_test_index as first columns from adverse_df
    shap_df.insert(0, 'instance_index', adverse_df['instance_index'].values)
    shap_df.insert(1, 'original_test_index', adverse_df['original_test_index'].values)
    # Add instance prediction_score column
    shap_df.insert(2, 'predicted_probability', adverse_df['prediction_score'].values)
        
    # Save SHAP values
    shap_output_path = os.path.join(dataset_path, f'{dataset_name}_shap.csv')
    shap_df.to_csv(shap_output_path, index=False)
    
    # ===== COUNTERFACTUALS (using NICE Algorithm) =====
    # Get features for counterfactual generation (excluding frozen features)
    cf_feature_cols = [col for col in feature_cols if col not in frozen_features]
    
    print(f"  Generating counterfactuals using {len(cf_feature_cols)} mutable features (excluding {len(frozen_features)} frozen)")
    
    # Get predictions for test set to identify good class instances
    test_pred = rf_model.predict(test_features)
    
    # Find instances with opposite class (good class, 0)
    good_class_indices = np.where(test_pred == 0)[0]
    good_class_instances = test_features.iloc[good_class_indices]
    
    # For NICE algorithm, use only mutable features for KNN fitting and distance calculation
    good_class_instances_mutable = good_class_instances[cf_feature_cols]
    
    # Fit KNN on good class instances for NICE algorithm (using only mutable features)
    all_counterfactuals = []
    
    if len(good_class_instances) > 0:
        nbrs = NearestNeighbors(n_neighbors=min(num_cf, len(good_class_instances)), 
                                algorithm='ball_tree').fit(good_class_instances_mutable.values)
        
        # Generate counterfactuals for each adverse instance
        for adverse_idx, (_, adverse_row) in enumerate(adverse_df.iterrows()):
            if (adverse_idx + 1) % max(1, len(adverse_df) // 10) == 0:
                print(f"    Processing instance {adverse_idx + 1}/{len(adverse_df)}")
            
            try:
                # Get ONLY mutable feature values for this adverse instance (for KNN query)
                instance_mutable = adverse_row[cf_feature_cols].values.reshape(1, -1)
                instance_idx_val = adverse_row['instance_index']
                original_idx_val = adverse_row['original_test_index']
                
                # Find nearest neighbors in good class using NICE (based on mutable features only)
                distances, indices = nbrs.kneighbors(instance_mutable)
                
                # Extract the nearest instances as counterfactuals
                for cf_idx, neighbor_idx in enumerate(indices[0]):
                    cf_instance = good_class_instances.iloc[neighbor_idx].copy()
                    
                    # Replace frozen features with values from the original adverse instance
                    for frozen_feat in frozen_features:
                        if frozen_feat in cf_instance.index:
                            cf_instance[frozen_feat] = adverse_row[frozen_feat]
                    
                    cf_instance['instance_index'] = instance_idx_val
                    cf_instance['original_test_index'] = original_idx_val
                    cf_instance['CF_number'] = cf_idx + 1
                    cf_instance['distance_to_original'] = distances[0][cf_idx]
                    all_counterfactuals.append(cf_instance)
                    
            except Exception as e:
                print(f"    Warning: Could not generate counterfactuals for instance {adverse_idx}: {str(e)}")
        
        print(f"  Generated {len(all_counterfactuals)} total counterfactuals using NICE")
        
        # Create counterfactual dataframe
        cf_df = pd.DataFrame(all_counterfactuals)
        
        # Drop protected attributes from counterfactual dataframe
        cf_df = cf_df.drop(columns=protected_attributes, errors='ignore')
        
        # Reorganize columns: instance_index and original_test_index first, then CF metadata
        # Ensure instance_index, original_test_index, and CF_number are integer types
        cf_df['instance_index'] = cf_df['instance_index'].astype(int)
        cf_df['original_test_index'] = cf_df['original_test_index'].astype(int)
        cf_df['CF_number'] = cf_df['CF_number'].astype(int)
        
        cols = ['instance_index', 'original_test_index', 'CF_number', 'distance_to_original']
        feature_cf_cols = [col for col in cf_df.columns if col not in cols]
        cf_df = cf_df[cols + feature_cf_cols]
        
        # Save counterfactuals
        cf_output_path = os.path.join(dataset_path, f'{dataset_name}_counterfactual.csv')
        cf_df.to_csv(cf_output_path, index=False)
        print(f"  Saved counterfactuals to {cf_output_path}")
    else:
        print(f"  Warning: No instances with good class (0) found for NICE algorithm")

def main():
    """Main function to generate all explanations and save to individual files."""
    
    for dataset_name, config in datasets.items():
        try:
            generate_explanations(dataset_name, config, NUM_COUNTERFACTUALS)
        except Exception as e:
            print(f"  ERROR processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":
    main()
