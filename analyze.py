import pandas as pd
import json
import os

def analyze_cf(name, cf_path, adv_path, json_path):
    print(f"--- {name} Analysis ---")
    df_cf = pd.read_csv(cf_path)
    df_adv = pd.read_csv(adv_path)
    
    # Identify distance column (could be "distance" or "distance_to_original")
    dist_col = "distance" if "distance" in df_cf.columns else "distance_to_original"
    
    # 1. Row count
    expected_rows = len(df_adv) * 3
    print(f"Total CF rows: {len(df_cf)} (Expected: {expected_rows})")
    
    # 2. CF_number check
    counts = df_cf["CF_number"].value_counts()
    print(f"CF_number counts: {counts.to_dict()}")
    
    # 3. Distance check
    print(f"Distance statistics ({dist_col}):\n{df_cf[dist_col].describe()}")
    
    # 4. Changes check
    # Exclude non-feature columns
    exclude = ["instance_index", "original_test_index", "CF_number", "distance", "distance_to_original", "predicted_label", "predicted_probability", "status", "bar_pass_prediction"]
    feature_cols = [c for c in df_adv.columns if c not in ["predicted_label", "predicted_probability"]]
    
    # Align rows
    changes = []
    for i, row in df_cf.iterrows():
        # Match based on index assuming 3 CFs per row in same order
        original_idx = i // 3
        original_row = df_adv.iloc[original_idx]
        diffs = 0
        for col in feature_cols:
            if col in row and col in original_row:
                if row[col] != original_row[col]:
                    diffs += 1
        changes.append(diffs)
    
    df_cf["num_changes"] = changes
    print(f"Changes per CF statistics:\n{df_cf['num_changes'].describe()}")
    
    # 5. JSON check
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"JSON instance count: {len(data)}")
    print(f"Sample JSON keys: {list(data.keys())[:2]}")
    if len(data) > 0:
        first_key = list(data.keys())[0]
        instance_data = data[first_key]
        print(f"Sample JSON content for {first_key}:\n{json.dumps(instance_data, indent=2)[:500]}...")

    print("\nSample CF rows (first 3):")
    # Take first 3 features for display
    disp_cols = feature_cols[:3]
    print(df_cf[["CF_number", dist_col, "num_changes"] + disp_cols].head(3))
    print("\n" + "="*30 + "\n")

analyze_cf("Credit", "datasets_prep/data/credit_dataset/credit_counterfactual.csv", "datasets_prep/data/credit_dataset/credit_adverse.csv", "datasets_prep/data/credit_dataset/credit_counterfactual_analysis.json")
analyze_cf("Law", "datasets_prep/data/law_dataset/law_counterfactual.csv", "datasets_prep/data/law_dataset/law_adverse.csv", "datasets_prep/data/law_dataset/law_counterfactual_analysis.json")
