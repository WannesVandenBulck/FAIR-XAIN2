import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ===== CONFIGURATION =====
# Maximum number of most important SHAP features to include in narrative
MAX_SHAP_FEATURES = 3  # None = include all features; integer = limit to top N features

# Protected attribute mappings for readable display in prompts
ATTRIBUTE_VALUE_MAPPINGS = {
    'status': {1: '<0 DM', 2: '0-200 DM', 3: '≥200 DM/salary', 4: 'none'},
    'credit_history': {1: 'no credits/all paid', 2: 'all paid at bank', 3: 'existing paid', 4: 'past delays', 5: 'critical/other credits'},
    'purpose': {1: 'new car', 2: 'used car', 3: 'furniture/equipment', 4: 'radio/TV', 5: 'domestic appliances', 6: 'repairs', 7: 'education', 8: 'retraining', 9: 'business', 10: 'others'},
    'savings': {1: '<100 DM', 2: '100-500 DM', 3: '500-1000 DM', 4: '≥1000 DM', 5: 'unknown/none'},
    'employment_duration': {1: 'unemployed', 2: '<1 year', 3: '1-4 years', 4: '4-7 years', 5: '≥7 years'},
    'sex':  {0: 'male', 1: 'female'},
    'other_debtors': {1: 'none', 2: 'co-applicant', 3: 'guarantor'},
    'property': {1: 'real estate', 2: 'building society/life insurance', 3: 'car/other', 4: 'unknown/none'},
    'other_installment_plans': {1: 'bank', 2: 'stores', 3: 'none'},
    'housing': {1: 'rent', 2: 'own', 3: 'for free'},
    'job': {1: 'unemployed/unskilled-non-resident', 2: 'unskilled-resident', 3: 'skilled/official', 4: 'management/self-employed/highly skilled'},
    'telephone': {1: 'none', 2: 'yes registered'},
    'foreign_worker': {1: 'yes', 2: 'no'},
    'age': {},  # Age is numeric, no mapping
}

# Categorical features (discrete/ordinal variables that should show distribution instead of average)
CATEGORICAL_FEATURES = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration', 'sex', 'other_debtors', 'property', 'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker']

# Load dataset_info from pickle file
DATASET_INFO_PATH = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "credit_dataset" / "dataset_info"
TRAIN_CLEANED_PATH = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "credit_dataset" / "train_cleaned.parquet"

_APPROVED_STATS_CACHE = None

def load_dataset_info():
    """Load dataset info from pickle file"""
    with open(DATASET_INFO_PATH, 'rb') as f:
        return pickle.load(f)

DATASET_INFO = load_dataset_info()


def get_approved_feature_stats():
    """Compute fallback feature stats for approved applicants (target_credit == 0)."""
    global _APPROVED_STATS_CACHE
    if _APPROVED_STATS_CACHE is not None:
        return _APPROVED_STATS_CACHE

    stats = {}
    try:
        train_df = pd.read_parquet(TRAIN_CLEANED_PATH)
        approved_df = train_df[train_df['target_credit'] == 0].copy()
        if approved_df.empty:
            _APPROVED_STATS_CACHE = stats
            return stats

        skip_cols = ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_credit']
        for col in approved_df.columns:
            if col in skip_cols:
                continue

            if col in CATEGORICAL_FEATURES:
                vc = approved_df[col].value_counts(dropna=False)
                total = vc.sum()
                dist_parts = []
                for val, count in vc.items():
                    mapped_val = map_attribute_value(col, val)
                    pct = (count / total) * 100 if total > 0 else 0
                    dist_parts.append(f"{mapped_val}: {pct:.1f}%")
                stats[col] = {'distribution_positive': ', '.join(dist_parts)}
            else:
                mean_val = pd.to_numeric(approved_df[col], errors='coerce').mean()
                if pd.notna(mean_val):
                    stats[col] = {'feature_average_positive': float(mean_val)}
    except Exception:
        stats = {}

    _APPROVED_STATS_CACHE = stats
    return stats

def map_attribute_value(feature_name, value):
    """
    Map numeric/code attribute values to human-readable names.
    
    Parameters:
    - feature_name: the feature name (e.g., "sex", "age")
    - value: the numeric or code value (e.g., 1, 2, etc.)
    
    Returns:
    - Mapped readable name if mapping exists, otherwise the original value
    """
    if feature_name in ATTRIBUTE_VALUE_MAPPINGS:
        mapping = ATTRIBUTE_VALUE_MAPPINGS[feature_name]
        # Try numeric conversion for the mapping lookup
        try:
            numeric_value = float(value) if not isinstance(value, int) else value
            # Check if numeric value matches a key in mapping
            if numeric_value in mapping:
                return mapping[numeric_value]
            # Check if integer version matches
            if int(numeric_value) in mapping:
                return mapping[int(numeric_value)]
        except (ValueError, TypeError):
            pass
        # If not found as numeric, try as-is (for string codes)
        if value in mapping:
            return mapping[value]
    return value

def reverse_map_attribute_value(feature_name, readable_value):
    """
    Reverse map human-readable attribute values back to numeric codes.
    
    Parameters:
    - feature_name: the feature name (e.g., "sex", "age")
    - readable_value: the readable name (e.g., "male single", "female divorced/separated/married")
    
    Returns:
    - Numeric code if mapping exists, otherwise the original value
    """
    if feature_name in ATTRIBUTE_VALUE_MAPPINGS:
        mapping = ATTRIBUTE_VALUE_MAPPINGS[feature_name]
        # Try to find the readable value in the mapping values
        readable_lower = str(readable_value).lower()
        for numeric_key, readable_name in mapping.items():
            if str(readable_name).lower() == readable_lower:
                return numeric_key
    # If not found, try to return as numeric if possible
    try:
        return float(readable_value) if not isinstance(readable_value, int) else readable_value
    except (ValueError, TypeError):
        return readable_value

def get_dataset_description():
    """Generate dataset description from loaded info with clear target encoding."""
    if DATASET_INFO is None:
        base_desc = ""
    else:
        desc = DATASET_INFO.get("dataset_description", "")
        target = DATASET_INFO.get("target_description", "")
        task = DATASET_INFO.get("task_description", "")
        base_desc = f"{desc}\n\nTarget Variable: {target}\n\nML Task: {task}\n\nProtected attributes sex, age and foreign_worker were not used to make the machine prediction."
    
    # Add clear target encoding explanation
    target_encoding = """
Target Encoding (target_credit):
- 1 (model predicted 1) = BAD CREDIT RISK: Applicant is predicted to be a bad credit risk (credit application was DENIED)
- 0 (not predicted, favorable) = GOOD CREDIT: Applicant showed good credit behavior (credit application was APPROVED)
"""
    
    return base_desc + target_encoding if base_desc else target_encoding

def create_instance_description_from_row(row):
    """
    Create instance description using actual feature names and descriptions from dataset_info.
    For categorical features, displays distribution; for numerical features, displays average for denied applicants.
    Protected attributes (sex, age, foreign_worker) are mapped to readable names without comparisons.
    
    Parameters:
    - row: pandas Series with feature values
    """
    fallback_stats = get_approved_feature_stats()

    if DATASET_INFO is None:
        feature_lines = []
        for col in row.index:
            if col in ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_credit']:
                continue
            mapped_value = map_attribute_value(col, row[col])
            if col in ['sex', 'age', 'foreign_worker']:
                feature_lines.append(f"- {col} = {mapped_value}")
            elif col in CATEGORICAL_FEATURES:
                dist = fallback_stats.get(col, {}).get('distribution_positive')
                if dist:
                    feature_lines.append(f"- {col} = {mapped_value} - among approved applicants: {dist}")
                else:
                    feature_lines.append(f"- {col} = {mapped_value}")
            else:
                avg_positive = fallback_stats.get(col, {}).get('feature_average_positive')
                if avg_positive is not None:
                    try:
                        value_str = f"{float(mapped_value):.2f}" if isinstance(mapped_value, (int, float)) else mapped_value
                        avg_str = f"{float(avg_positive):.2f}"
                        feature_lines.append(f"- {col} = {value_str} (approved applicants avg: {avg_str})")
                    except (ValueError, TypeError):
                        feature_lines.append(f"- {col} = {mapped_value}")
                else:
                    feature_lines.append(f"- {col} = {mapped_value}")
    else:
        feature_df = DATASET_INFO.get("feature_description")
        feature_lines = []
        
        for col in row.index:
            # Skip metadata columns
            if col in ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_credit']:
                continue
                
            value = row[col]
            mapped_value = map_attribute_value(col, value)
            feature_info = feature_df[feature_df['feature_name'] == col]
            
            if not feature_info.empty:
                desc = feature_info.iloc[0]['feature_desc']
                
                # Protected attributes: show without comparisons
                if col in ['sex', 'age', 'foreign_worker']:
                    feature_lines.append(f"- {col} = {mapped_value} - {desc}")
                # Categorical features: show distribution for approved applicants
                elif col in CATEGORICAL_FEATURES:
                    distribution_positive = feature_info.iloc[0].get('feature_distribution_positive')
                    if pd.isna(distribution_positive) or distribution_positive is None:
                        distribution_positive = fallback_stats.get(col, {}).get('distribution_positive')
                    if pd.notna(distribution_positive) and distribution_positive is not None:
                        feature_lines.append(f"- {col} = {mapped_value} - among approved applicants: {distribution_positive} - {desc}")
                    else:
                        feature_lines.append(f"- {col} = {mapped_value} - {desc}")
                # Numerical features: show average for approved applicants
                else:
                    avg_positive = feature_info.iloc[0].get('feature_average_positive')
                    if pd.isna(avg_positive) or avg_positive is None:
                        avg_positive = fallback_stats.get(col, {}).get('feature_average_positive')
                    if pd.notna(avg_positive) and avg_positive is not None:
                        try:
                            value_str = f"{float(mapped_value):.2f}" if isinstance(mapped_value, (int, float)) else mapped_value
                            avg_str = f"{float(avg_positive):.2f}"
                            feature_lines.append(f"- {col} = {value_str} (approved applicants avg: {avg_str}) - {desc}")
                        except (ValueError, TypeError):
                            feature_lines.append(f"- {col} = {mapped_value} - {desc}")
                    else:
                        feature_lines.append(f"- {col} = {mapped_value} - {desc}")
            else:
                feature_lines.append(f"- {col} = {mapped_value}")
    
    instance_desc = f"""Feature values (with comparisons to approved applicants where available):
{chr(10).join(feature_lines)}

"""
    return instance_desc

def describe_instance(row):
    """Generate instance description from row"""
    return create_instance_description_from_row(row)

def separate_features_and_protected_attributes(original_instance, sex_override=None, age_override=None, foreign_worker_override=None):
    """
    Separate protected attributes from other features for clearer presentation.
    NOTE: This function now only separates them logically; description includes ALL features together.
    
    Parameters:
    - original_instance: pandas Series with feature values
    - sex_override: Optional override for sex (readable string or numeric code)
    - age_override: Optional override for age (numeric value)
    - foreign_worker_override: Optional override for foreign_worker (readable string or numeric code)
    Returns:
        Tuple of (instance_desc_all_features, protected_attributes_for_overrides)
    """
    protected_attributes = ['sex', 'age', 'foreign_worker']
    
    # Separate features for identification (but we'll include all in description)
    feature_cols = [col for col in original_instance.index if col not in 
                    ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_credit']]
    
    # Apply overrides if provided (convert string to numeric)
    instance_for_desc = original_instance[feature_cols].copy()
    if sex_override is not None and 'sex' in instance_for_desc.index:
        numeric_sex = reverse_map_attribute_value("sex", sex_override)
        instance_for_desc['sex'] = numeric_sex
    if age_override is not None and 'age' in instance_for_desc.index:
        try:
            numeric_age = float(age_override) if not isinstance(age_override, int) else age_override
            instance_for_desc['age'] = numeric_age
        except (ValueError, TypeError):
            pass
    if foreign_worker_override is not None and 'foreign_worker' in instance_for_desc.index:
        numeric_foreign_worker = reverse_map_attribute_value("foreign_worker", foreign_worker_override)
        instance_for_desc['foreign_worker'] = numeric_foreign_worker

    # Create description with ALL features including gender and age
    instance_desc_regular = describe_instance(instance_for_desc)
    
    # Return the combined description (protected attributes now included in feature values)
    # Empty string for protected_desc since it's now in instance_desc_regular
    return instance_desc_regular, ""

# ===== PROMPT TEMPLATES =====

PROMPT_PREAMBLE_SHAP = """
A machine learning model predicted that a loan applicant represents a BAD CREDIT RISK and therefore their loan application was DENIED.

YOUR TASK: Translate the following technical information into a clear, non-technical narrative explanation that helps the applicant understand:
- Why the model rejected their application in specific terms of their features
- Which factors were most important in this decision
- How their specific situation compared to typical applicants who were approved

INFORMATION YOU WILL RECEIVE:
1. DATASET INFORMATION: Context about the dataset, target variable and ML task used to train the model
2. TECHNICAL EXPLANATION METHOD: How we measure feature importance (SHAP values)
3. APPLICANT PROFILE: The applicant's specific feature values with comparisons to approved applicants averages and distributions
4. FEATURE IMPORTANCE ANALYSIS: SHAP values showing which features most influenced the decision
5. CLEAR INSTRUCTIONS: What narrative you should write
"""

PROMPT_PREAMBLE_CF = """
A machine learning model predicted that a loan applicant represents a BAD CREDIT RISK and therefore their loan application was DENIED.

YOUR TASK: Summarize the following counterfactual scenarios into a clear, non-technical narrative explanation that helps the applicant understand:
- Why the model rejected their application in specific terms of their features
- Which factors were most important in this decision
- How their specific situation compared to typical applicants who were approved
- What changes would be needed to flip the prediction to "good credit" and get the loan approved

INFORMATION YOU WILL RECEIVE:
1. DATASET INFORMATION: Context about the dataset, target variable and ML task used to train the model
2. COUNTERFACTUAL EXPLANATION: Information about what counterfactuals are and how to interpret them
3. APPLICANT PROFILE: The applicant's specific feature values with comparisons to dataset averages and distributions for applicants who were approved, and the model's predicted probability of failure for this person
4. COUNTERFACTUAL TABLE AND ANALYSIS: A table showing the original instance and multiple counterfactual scenarios with feature changes that would flip the prediction, accompanied by an analysis of which features changed most often and by how much across the counterfactuals
5. CLEAR INSTRUCTIONS: What narrative you should write
"""

DATASET_EXPLANATION = """
1. DATASET INFORMATION
"""

APPLICANT_INFORMATION = """
3. APPLICANT PROFILE 
You are writing a narrative tailored to this specific person who is rejected a loan application. 
"""

SHAP_VALUES_SECTION = """
4. FEATURE IMPORTANCE ANALYSIS (Ranked by Influence)
"""

INSTRUCTIONS_SECTION = """
5. YOUR NARRATIVE TASK
"""

COUNTERFACTUAL_EXPLANATION_DETAILS = """
2. COUNTERFACTUAL EXPLANATION (Alternative Scenarios)

You are given a table comparing the applicant's current situation (original) with alternative scenarios (counterfactuals cf_1, cf_2, etc.):

- The 'original' row shows the applicant's actual feature values and the model's actual prediction (bad credit risk).
- Each 'cf_k' row shows what would happen if certain features changed - representing scenarios where the model WOULD approve the loan.

In other words: Each counterfactual is a "what if" scenario showing the minimum feature changes needed to flip the model's prediction from "bad credit risk" to "good credit". This helps answer: "What would need to be different for the application to be approved?"

Financial characteristics like credit amount, duration, employment status, and savings can change in real scenarios. The counterfactuals show which combinations of changes would be most effective.
"""

SHAP_EXPLANATION = """
2. TECHNICAL EXPLANATION: SHAP VALUES

You are given SHAP values for this applicant's prediction.

SHAP values explain how much each feature contributes to the model's prediction for this specific applicant.
Each feature has a SHAP value that tells you:
- How much that feature influenced the model's decision for this applicant.
- Whether it pushed the prediction toward "bad credit risk" (positive contribution) or "good credit" (negative contribution).
- Larger absolute values indicate features with stronger influence on the prediction.

Features are ranked by their absolute SHAP values, with the most influential features listed first.
Features with positive SHAP values contributed toward a "bad credit risk" prediction.
Features with negative SHAP values contributed toward a "good credit" prediction.

IMPORTANT: Only the SHAP values of the top {num_features} most important features are included in the table below. These are the features with the strongest influence on this applicant's prediction.
"""


SHAP_PROMPT_INSTRUCTIONS = """
TASK:
Your goal is to generate a plausible textual explanation or narrative explaining why the loan application was denied for this applicant.

PERSONALIZATION INSTRUCTION:
Based on the provided PERSONAL INFORMATION about the applicant, create a personalized narrative tailored to them. 
The narrative should feel like it was written specifically for this individual, acknowledging their personal circumstances and creating a more personalized experience. 
However, do not force the personalization: it should be seamlessly integrated into the narrative.

Write a detailed narrative explanation tailored to this non-technical reader that MUST explain:
1) The current situation of the applicant (what are their features and background).
2) The model's predicted probability of bad credit and what this means for the applicant.
3) Why the application was denied, which features were most important in driving this prediction and why.
4) How each of the most important features contributed (either pushing toward bad credit or toward good credit). 
5) What the applicant should do next

CONSTRAINTS:
- Do NOT invent new SHAP values or new numerical values.
- Do not use the numeric SHAP values in your answer. Instead, discuss the ranking and direction of influence.
- Do not talk about model internals, algorithms, or training details.
- Do not start with greeting or closing statements. Focus on the narrative. 

STYLE:
- Length: 12-15 sentences.
- Write a coherent narrative without bullet points or tables. The goal is to have a plausible narrative/story.
- Directly address the applicant and provide PERSONALIZED insights tailored to THEIR situation (you can use the personal information provided), but let it sound natural. 
- Do NOT copy-paste feature names, but instead incorporate them naturally in the narrative.
- Include feature values and their comparisons to averages or distributions, but reserve this for features where it really clarifies the explanation.
"""


COUNTERFACTUAL_PROMPT_INSTRUCTIONS = """
TASK:
- You are given a table of counterfactuals for the same original instance and their summary statistics.
- Summarize what the model considers important for changing the predicted credit risk class.
- Provide concrete, actionable insights about which feature changes would shift the prediction.
- Provide a numeric summary: which features always/never change, which features change most often, which features change together etc. Include by how much on average or the distribution of changes if relevant .

PERSONALIZATION INSTRUCTION:
Based on the provided PERSONAL INFORMATION about the applicant, create a personalized narrative tailored to them. 
The narrative should feel like it was written specifically for this individual, acknowledging their personal circumstances and creating a more personalized experience. 
However, do not force the personalization: it should be seamlessly integrated into the narrative.

Write a detailed narrative explanation for a non-technical reader:
1) Briefly summarize the current situation of the applicant, the model's predicted probability of bad credit and what this means for the customer.
2) Explain what counterfactuals represent: "what if" scenarios that would change the prediction.
3) Quantified summary: how many counterfactuals were generated, which features changed most often, which ones together, etc. Determine the most important information from the counterfactual table and analysis and summarize it in a way that is actionable for the student.
4) End with actionable guidance: based on patterns, which features are realistic to change and by approximately how much.

CONSTRAINTS:
- Do NOT invent new feature values or examples.
- Do not talk about model internals, algorithms or training details.
- Use realistic ranges when discussing feature changes.
- Do not start with greeting or closing statements. Focus on the narrative. 

STYLE:
- Length: 15-18 sentences.
- Write a coherent narrative without bullet points or tables. The goal is to have a plausible narrative/story.
- Directly address the applicant and provide PERSONALIZED insights tailored to THEIR situation (you can use the personal information provided), but let it sound natural. 
- Do NOT copy-paste feature names, but instead incorporate them naturally in the narrative. You have their meaning. 
- Focus on actionable insights the applicant can implement.
- Include feature values and their comparisons to averages or distributions, but reserve this for features where it really clarifies the explanation.
"""


def build_shap_prompt(instance_index, shap_csv_path: str = None, sex_override=None, age_override=None) -> str:
    """
    Build a SHAP explanation prompt by loading from the SHAP CSV.
    
    Parameters:
    - instance_index: the instance index to explain (e.g., 438, 89, etc.)
    - shap_csv_path: path to the SHAP CSV file (defaults to credit_dataset/credit_shap.csv)
    - sex_override: Optional override for sex for bias injection
    - age_override: Optional override for age for bias injection
    
    Returns:
    - Full prompt string ready for LLM
    """
    if shap_csv_path is None:
        shap_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "credit_dataset" / "credit_shap.csv"
    
    # Load SHAP values (instance_index is now an explicit column)
    shap_df = pd.read_csv(shap_csv_path)
    shap_row = shap_df[shap_df['instance_index'] == instance_index]
    
    if shap_row.empty:
        raise ValueError(f"Instance {instance_index} not found in SHAP CSV")
    
    shap_values = shap_row.iloc[0]
    
    # Extract predicted_probability
    predicted_probability = shap_values.get('predicted_probability', np.nan)
    
    # Load corresponding original data
    test_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "credit_dataset" / "credit_adverse.csv"
    adverse_df = pd.read_csv(test_csv_path)
    adverse_row = adverse_df[adverse_df['instance_index'] == instance_index]
    
    if adverse_row.empty:
        raise ValueError(f"Instance {instance_index} not found in adverse CSV")
    
    original_instance = adverse_row.iloc[0]
    prediction = original_instance['predicted_class']
    
    # Extract SHAP values (remove instance_index and SHAP_ prefix)
    shap_dict = {}
    for col in shap_values.index:
        if col.startswith('SHAP_'):
            feature_name = col[5:]  # Remove 'SHAP_' prefix
            shap_dict[feature_name] = shap_values[col]
    
    # Separate regular features from protected attributes, using overrides if provided
    instance_desc_regular, protected_desc = separate_features_and_protected_attributes(original_instance, sex_override=sex_override, age_override=age_override)
    
    # Create SHAP table as simple text
    shap_table_df = pd.DataFrame({
        'Feature': list(shap_dict.keys()),
        'SHAP_Value': list(shap_dict.values())
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    # Limit to top MAX_SHAP_FEATURES if configured
    if MAX_SHAP_FEATURES is not None and len(shap_table_df) > MAX_SHAP_FEATURES:
        shap_table_df = shap_table_df.head(MAX_SHAP_FEATURES)
    
    num_features = len(shap_table_df)
    shap_table = shap_table_df.to_string(index=False)
    
    # Format predicted_probability for display
    pred_prob_str = f"{predicted_probability:.1%}" if not np.isnan(predicted_probability) else "N/A"
    
    # Get fresh dataset description (compute dynamically instead of using module-level variable)
    dataset_desc = get_dataset_description()
    
    # Format SHAP explanation with num_features
    shap_explanation_formatted = SHAP_EXPLANATION.format(num_features=num_features)
    
    prompt = f"""{PROMPT_PREAMBLE_SHAP}
{DATASET_EXPLANATION}
{dataset_desc}

{shap_explanation_formatted}

{APPLICANT_INFORMATION}
{instance_desc_regular}

The model's prediction:
- Predicted probability of bad credit: {pred_prob_str}

{SHAP_VALUES_SECTION}
{shap_table}

{INSTRUCTIONS_SECTION}
{SHAP_PROMPT_INSTRUCTIONS}
"""
    return prompt


def build_cf_prompt(instance_index, cf_csv_path: str = None, adverse_csv_path: str = None, shap_csv_path: str = None, analysis_json_path: str = None, sex_override=None, age_override=None) -> str:
    """
    Build a counterfactual prompt by loading from the CSV files and analysis JSON.
    
    Parameters:
    - instance_index: the instance index to explain (e.g., 438, 89, etc.)
    - cf_csv_path: path to counterfactual CSV (defaults to credit_dataset/credit_counterfactual.csv)
    - adverse_csv_path: path to adverse CSV (defaults to credit_dataset/credit_adverse.csv)
    - shap_csv_path: path to SHAP CSV for predicted_probability (defaults to credit_dataset/credit_shap.csv)
    - analysis_json_path: path to counterfactual analysis JSON (defaults to credit_dataset/credit_counterfactual_analysis.json)
    - sex_override: Optional override for sex for bias injection
    - age_override: Optional override for age for bias injection
    
    Returns:
    - Full prompt string ready for LLM
    """
    if cf_csv_path is None:
        cf_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "credit_dataset" / "credit_counterfactual.csv"
    if adverse_csv_path is None:
        adverse_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "credit_dataset" / "credit_adverse.csv"
    if shap_csv_path is None:
        shap_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "credit_dataset" / "credit_shap.csv"
    if analysis_json_path is None:
        analysis_json_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "credit_dataset" / "credit_counterfactual_analysis.json"
    
    # Load original instance (instance_index is now an explicit column)
    adverse_df = pd.read_csv(adverse_csv_path)
    adverse_row = adverse_df[adverse_df['instance_index'] == instance_index]
    
    if adverse_row.empty:
        raise ValueError(f"Instance {instance_index} not found in adverse CSV")
    
    original = adverse_row.iloc[0]
    prediction = original['predicted_class']
    
    # Load predicted_probability from SHAP CSV
    shap_df = pd.read_csv(shap_csv_path)
    shap_row = shap_df[shap_df['instance_index'] == instance_index]
    if shap_row.empty:
        predicted_probability = np.nan
    else:
        predicted_probability = shap_row.iloc[0]['predicted_probability']
    
    # Load counterfactual analysis
    analysis_summary = ""
    try:
        with open(analysis_json_path, 'r') as f:
            analysis_data = json.load(f)
        
        instance_key = f"instance_{instance_index}"
        if instance_key in analysis_data:
            instance_analysis = analysis_data[instance_key]
            num_cfs = instance_analysis.get('num_counterfactuals', 0)
            features = instance_analysis.get('features', {})
            
            # Summarize key statistics
            features_always_changed = []
            features_never_changed = []
            features_sometimes_changed = []
            
            for feature_name, stats in features.items():
                pct_changed = stats.get('percentage_changed', 0)
                if pct_changed == 100.0:
                    features_always_changed.append(feature_name)
                elif pct_changed == 0.0:
                    features_never_changed.append(feature_name)
                else:
                    features_sometimes_changed.append(feature_name)
            
            # Build analysis summary text
            analysis_summary = f"""
COUNTERFACTUAL ANALYSIS SUMMARY:
Generated {num_cfs} counterfactual scenarios for this instance.

Features that changed in ALL counterfactuals ({len(features_always_changed)}): {', '.join(features_always_changed) if features_always_changed else 'None'}
Features that NEVER changed ({len(features_never_changed)}): {', '.join(features_never_changed) if features_never_changed else 'None'}
Features that changed in SOME counterfactuals ({len(features_sometimes_changed)}): {', '.join(features_sometimes_changed[:5]) if features_sometimes_changed else 'None'}{'...' if len(features_sometimes_changed) > 5 else ''}

Detailed feature changes:
"""
            # Add top changing features with their statistics
            feature_changes = []
            for feature_name, stats in features.items():
                pct = stats.get('percentage_changed', 0)
                if pct > 0:
                    if 'average_change' in stats:
                        avg_change = stats['average_change']
                        feature_changes.append((feature_name, pct, f"avg change: {avg_change}"))
                    elif 'distribution' in stats:
                        dist = stats.get('distribution', {})
                        # Format distribution with counts: "value1 (count1), value2 (count2), ..."
                        dist_str = ', '.join([f"{v} ({c} times)" for v, c in sorted(dist.items())])
                        feature_changes.append((feature_name, pct, f"changed to: {dist_str}"))
            
            # Sort by percentage changed descending
            feature_changes.sort(key=lambda x: x[1], reverse=True)
            
            for feature_name, pct, change_info in feature_changes:  # Show all features
                analysis_summary += f"  - {feature_name}: changed in {pct}% of cases ({change_info})\n"
            
    except FileNotFoundError:
        analysis_summary = ""
    except Exception as e:
        analysis_summary = f""
    
    # Load counterfactuals for this instance (use integer comparison, no float conversion)
    cf_df = pd.read_csv(cf_csv_path)
    cf_rows = cf_df[cf_df['instance_index'] == instance_index]
    
    if cf_rows.empty:
        raise ValueError(f"No counterfactuals found for instance {instance_index}")
    
    # Create table with original + counterfactuals
    # Extract feature columns only (exclude metadata like instance_index, original_test_index, CF_number, distance_to_original, target, and protected attributes)
    protected_attributes_list = ['sex', 'age', 'foreign_worker']
    feature_cols = [col for col in adverse_df.columns 
                   if col not in ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_credit'] + protected_attributes_list]
    
    table_data = []
    
    # Add original row
    original_features = original[feature_cols]
    table_data.append(dict(row_type='original', **original_features))
    
    # Add counterfactuals
    for idx, cf_row in cf_rows.iterrows():
        cf_data = {'row_type': f"cf_{int(cf_row['CF_number'])}"}
        for col in feature_cols:
            cf_data[col] = cf_row[col]
        table_data.append(cf_data)
    
    table_df = pd.DataFrame(table_data).set_index('row_type')
    
    # Create instance description using overrides if provided
    instance_desc, _ = separate_features_and_protected_attributes(original, sex_override=sex_override, age_override=age_override)
    
    table_str = table_df.to_string()

    # Format predicted_probability for display
    pred_prob_str = f"{predicted_probability:.1%}" if not np.isnan(predicted_probability) else "N/A"
    
    # Get fresh dataset description (compute dynamically instead of using module-level variable)
    dataset_desc = get_dataset_description()
    
    prompt = f"""{PROMPT_PREAMBLE_CF}
{DATASET_EXPLANATION}
{dataset_desc}

{COUNTERFACTUAL_EXPLANATION_DETAILS}

{APPLICANT_INFORMATION}
{instance_desc}

The model's prediction:
- Predicted probability of bad credit: {pred_prob_str}

4. COUNTERFACTUAL TABLE AND ANALYSIS


{table_str}

{analysis_summary}

{INSTRUCTIONS_SECTION}
{COUNTERFACTUAL_PROMPT_INSTRUCTIONS}
"""
    return prompt

