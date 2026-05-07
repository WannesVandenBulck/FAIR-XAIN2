import pandas as pd
import numpy as np
import pickle
import os
import shap

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
    },
    'saudi': {
        'path': r'datasets_prep/data/saudi_dataset',
        'target_col': 'target_saudi',
        'adverse_file': 'saudi_adverse.csv',
        'protected_attrs': ['Gender', 'Age', 'Health_Issues']
    },
    'student': {
        'path': r'datasets_prep/data/student_dataset',
        'target_col': 'target_student',
        'adverse_file': 'student_adverse.csv',
        'protected_attrs': ['sex', 'age', 'health']
    }
}

def generate_shap_values(dataset_name, config):
    dataset_path = config['path']
    target_col = config['target_col']
    protected_attrs = config['protected_attrs']
    
    print(f'\nProcessing {dataset_name}...')
    
    adverse_df = pd.read_csv(os.path.join(dataset_path, config['adverse_file']))
    print(f'  Loaded {len(adverse_df)} adverse instances')
    
    with open(os.path.join(dataset_path, 'RF.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    
    train_df = pd.read_parquet(os.path.join(dataset_path, 'train_cleaned.parquet'))
    train_df = train_df.drop(columns=protected_attrs, errors='ignore')
    feature_cols = train_df.drop(columns=[target_col]).columns
    
    print(f'  Computing SHAP values...')
    adverse_features = adverse_df[feature_cols]
    shap_values = shap.TreeExplainer(rf_model).shap_values(adverse_features)
    shap_values = shap_values[:, :, 1] if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 else (shap_values[1] if isinstance(shap_values, list) else shap_values)
    
    shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{col}' for col in feature_cols])
    shap_df.insert(0, 'instance_index', adverse_df['instance_index'])
    shap_df.insert(1, 'original_test_index', adverse_df['original_test_index'])
    shap_df.insert(2, 'predicted_probability', adverse_df['prediction_score'])
    
    output_path = os.path.join(dataset_path, f'{dataset_name}_shap.csv')
    shap_df.to_csv(output_path, index=False)
    print(f'  Saved SHAP values to {output_path}')

def main():
    for name, config in datasets.items():
        try:
            generate_shap_values(name, config)
        except Exception as e:
            print(f'  ERROR: {str(e)}')

if __name__ == "__main__":
    main()
