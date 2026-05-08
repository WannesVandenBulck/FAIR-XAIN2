import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path

# configuration 
# Maximum number of most important SHAP features to include in narrative
MAX_SHAP_FEATURES = 3  # None = include all features; integer = limit to top N features

# Protected attribute mappings for readable display in prompts
ATTRIBUTE_VALUE_MAPPINGS = {
    'Gender': {0: 'Female', 1: 'Male'},
    'Age': {0: '21-30', 1: '31-40', 3: '41+'},
    'Health_Issues': {0: 'no', 1: 'yes'},
    'Academic_degree': {0: 'secondary school', 1: 'bachelor', 2: 'master', 3: 'PhD'},
    'Years_Experience': {0: '1-5 years', 1: '6-10 years', 2: '11+ years'},
    'Years_experience_lastorganization': {0: '1-5 years', 1: '6-10 years', 2: '11+ years'},
    'Sector': {1: 'other', 2: 'medical', 3: 'education', 4: 'financial', 5: 'food'},
    'MonthlySalary': {0: '1k-5k SAR', 1: '6k-10k SAR', 2: '11k-15k SAR', 3: '16k+ SAR'},
    'MedicalInsurance': {0: 'no', 1: 'yes'},
    'Bonus': {0: 'no', 1: 'yes'},
    'OverTime': {0: 'no', 1: 'yes'},
    'Payment_Overtime': {0: 'no overtime', 1: 'no', 2: 'yes'},
    'Rewards_Wages_Satisfaction': {0: 'no', 1: 'yes'},
    'Get_Deserved_Promotion': {0: 'no', 1: 'yes'},
    'Training_programs_During_last_three_years': {0: 'none', 1: '1-3', 2: '4-6', 3: '7+'},
    'Useful_Training_Programs': {0: 'no', 1: 'yes'},
    'Business_Travel': {0: 'never', 1: 'rarely', 2: 'frequently'},
    'Job_Support': {0: 'low', 1: 'medium', 2: 'high'},
    'Recognition': {0: 'no', 1: 'yes'},
    'Emotional_Commitment': {0: 'low', 1: 'medium', 2: 'high'},
    'Job_Engagement': {0: 'easy', 1: 'medium', 2: 'difficult'},
    'Distance_to_work': {0: 'close', 1: 'medium', 2: 'far'},
    'Work_Live_Balance': {0: 'easy', 1: 'medium', 2: 'difficult'},
    'Physical_Stress': {0: 'no', 1: 'sometimes', 2: 'yes'},
    'Psychological_Exhaustion': {0: 'no', 1: 'sometimes', 2: 'yes'},
    'Job_Stability': {0: 'no', 1: 'yes'},
    'Environment_Satisfaction': {0: 'low', 1: 'medium', 2: 'high'},
    'Job_Satisfaction': {0: 'not satisfied', 1: 'satisfied', 2: 'very satisfied'},
    'Job_Opportunities': {0: 'no', 1: 'yes'},
}

# All features are categorical (ordinal categorical - should show distributions for employees who stayed)
CATEGORICAL_FEATURES = ['Gender', 'Age', 'Academic_degree', 'Sector', 'MonthlySalary', 
                        'MedicalInsurance', 'Bonus', 'OverTime', 'Payment_Overtime', 
                        'Rewards_Wages_Satisfaction', 'Get_Deserved_Promotion', 'Useful_Training_Programs',
                        'Business_Travel', 'Job_Support', 'Recognition', 'Emotional_Commitment',
                        'Job_Engagement', 'Distance_to_work', 'Work_Live_Balance', 'Physical_Stress',
                        'Psychological_Exhaustion', 'Job_Stability', 'Health_Issues', 'Environment_Satisfaction',
                        'Job_Satisfaction', 'Job_Opportunities']

# Load dataset_info from pickle file
DATASET_INFO_PATH = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "dataset_info"
TRAIN_CLEANED_PATH = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "train_cleaned.parquet"

_APPROVED_STATS_CACHE = None

def load_dataset_info():
    """Load dataset info from pickle file"""
    with open(DATASET_INFO_PATH, 'rb') as f:
        return pickle.load(f)

DATASET_INFO = load_dataset_info()


def get_approved_feature_stats():
    """Compute fallback feature stats for employees who stayed (target_saudi == 0)."""
    global _APPROVED_STATS_CACHE
    if _APPROVED_STATS_CACHE is not None:
        return _APPROVED_STATS_CACHE

    stats = {}
    try:
        train_df = pd.read_parquet(TRAIN_CLEANED_PATH)
        approved_df = train_df[train_df['target_saudi'] == 0].copy()
        if approved_df.empty:
            _APPROVED_STATS_CACHE = stats
            return stats

        skip_cols = ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_saudi', 'Gender', 'Age', 'Health_Issues']
        for col in approved_df.columns:
            if col in skip_cols:
                continue

            # All features: compute distribution
            vc = approved_df[col].value_counts(dropna=False)
            total = vc.sum()
            dist_parts = []
            for val, count in vc.items():
                mapped_val = map_attribute_value(col, val)
                pct = (count / total) * 100 if total > 0 else 0
                dist_parts.append(f"{mapped_val}: {pct:.1f}%")
            stats[col] = {'distribution_positive': ', '.join(dist_parts)}
    except Exception:
        stats = {}

    _APPROVED_STATS_CACHE = stats
    return stats

def map_attribute_value(feature_name, value):
    """
    Map numeric/code attribute values to human-readable names.
    
    Parameters:
    - feature_name: the feature name (e.g., "Gender", "Age")
    - value: the numeric or code value (e.g., 0, 1, etc.)
    
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
    - feature_name: the feature name (e.g., "Gender", "Age")
    - readable_value: the readable name (e.g., "Male", "21-30")
    
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
    desc = DATASET_INFO.get("dataset_description", "")
    target = DATASET_INFO.get("target_description", "")
    task = DATASET_INFO.get("task_description", "")
    base_desc = f"{desc}\n\nTarget Variable: {target}\n\nML Task: {task}\n\nProtected attributes Gender, Age and Health_Issues were not used to make the machine prediction."
    
    return base_desc 

def create_instance_description_from_row(row):
    """
    Create instance description using actual feature names and descriptions from dataset_info.
    All features are categorical and display distribution among employees who stayed.
    Protected attributes (Gender, Age, Health_Issues) are mapped to readable names without comparisons.
    
    Parameters:
    - row: pandas Series with feature values
    """
    fallback_stats = get_approved_feature_stats()
    feature_df = DATASET_INFO.get("feature_description")
    feature_lines = []
    
    for col in row.index:
        # Skip metadata columns
        if col in ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_saudi']:
            continue
            
        value = row[col]
        mapped_value = map_attribute_value(col, value)
        feature_info = feature_df[feature_df['feature_name'] == col]
        
        if not feature_info.empty:
            desc = feature_info.iloc[0]['feature_desc']
            
            # Protected attributes: show without comparisons
            if col in ['Gender', 'Age', 'Health_Issues']:
                feature_lines.append(f"- {col} = {mapped_value} ({desc})")
            # All other features: show distribution for employees who stayed
            else:
                distribution_positive = feature_info.iloc[0].get('feature_distribution_positive')
                if pd.isna(distribution_positive) or distribution_positive is None:
                    distribution_positive = fallback_stats.get(col, {}).get('distribution_positive')
                if pd.notna(distribution_positive) and distribution_positive is not None:
                    feature_lines.append(f"- {col} = {mapped_value} ({desc}) - distribution: {distribution_positive}")
                else:
                    feature_lines.append(f"- {col} = {mapped_value} ({desc})")
        else:
            feature_lines.append(f"- {col} = {mapped_value}")
    
    instance_desc = f"""Feature values (with comparisons to employees who were predicted to stay - showing distribution among employees who were predicted to stay):
{chr(10).join(feature_lines)}

"""
    return instance_desc

def describe_instance(row):
    """Generate instance description from row"""
    return create_instance_description_from_row(row)

def separate_features_and_protected_attributes(original_instance):
    """
    Separate protected attributes from other features for clearer presentation.
    Note: All features including protected attributes are included in the description.
    
    The actual attribute values come from the data CSV (which may be batch-specific
    for fairness evaluation with modified attributes).
    
    Parameters:
    - original_instance: pandas Series with feature values
    
    Returns:
        Tuple of (instance_desc_all_features, protected_attributes_for_info)
    """
    protected_attributes = ['Gender', 'Age', 'Health_Issues']
    
    # Separate features for identification (but we'll include all in description)
    feature_cols = [col for col in original_instance.index if col not in 
                    ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_saudi']]
    
    # Use instance data as-is (no overrides - they're in the CSV)
    instance_for_desc = original_instance[feature_cols].copy()
    
    # Create description with ALL features including protected attributes
    instance_desc_regular = describe_instance(instance_for_desc)
    
    # Return the combined description (protected attributes now included in feature values)
    # Empty string for protected_desc since it's now in instance_desc_regular
    return instance_desc_regular, ""

# prompt template SHAP only 

PROMPT_PREAMBLE_SHAP = """
A machine learning model predicted that an employee will LEAVE their job and is therefore DENIED a promotion.

YOUR TASK: Translate the following technical information into a clear, non-technical narrative explanation that helps the employee understand:
- Why the model predicted they will leave in specific terms of their features
- Which factors were most important in this decision
- How their specific situation compared to typical employees who were predicted to stay

INFORMATION YOU WILL RECEIVE:
1. DATASET INFORMATION: Context about the dataset, target variable and ML task used to train the model
2. TECHNICAL EXPLANATION METHOD: How we measure feature importance (SHAP values)
3. EMPLOYEE PROFILE: The employee's specific feature values with comparisons to employees who were predicted to stay distributions
4. FEATURE IMPORTANCE ANALYSIS: SHAP values showing which features most influenced the decision
5. CLEAR INSTRUCTIONS: What narrative you should write
"""

DATASET_EXPLANATION = """
1. DATASET INFORMATION
"""

APPLICANT_INFORMATION = """
3. EMPLOYEE PROFILE 
You are writing a narrative tailored to this specific person predicted to leave their job. 
"""

SHAP_VALUES_SECTION = """
4. FEATURE IMPORTANCE ANALYSIS (Ranked by Influence)
"""

INSTRUCTIONS_SECTION = """
5. YOUR NARRATIVE TASK
"""

SHAP_EXPLANATION = """
2. TECHNICAL EXPLANATION: SHAP VALUES

You are given SHAP values for this employee's prediction.

SHAP values explain how much each feature contributes to the model's prediction for this specific employee.
Each feature has a SHAP value that tells you:
- How much that feature influenced the model's decision for this employee.
- Whether it pushed the prediction toward "will leave" (positive contribution) or "will stay" (negative contribution).
- Larger absolute values indicate features with stronger influence on the prediction.

Features are ranked by their absolute SHAP values, with the most influential features listed first.
Features with positive SHAP values contributed toward a "will leave" prediction.
Features with negative SHAP values contributed toward a "will stay" prediction.

IMPORTANT: Only the SHAP values of the top {num_features} most important features are included in the table below. These are the features with the strongest influence on this employee's prediction.
"""


SHAP_PROMPT_INSTRUCTIONS = """
TASK:
Your goal is to generate a plausible textual explanation or narrative explaining why the employee is predicted to leave their job and is therefore denied a promotion.

PERSONALIZATION INSTRUCTION:
Based on the provided PERSONAL INFORMATION about the employee, create a personalized narrative tailored to them. 
The narrative should feel like it was written specifically for this individual, acknowledging their personal circumstances and creating a more personalized experience. 
However, do not force the personalization: it should be seamlessly integrated into the narrative.

Write a detailed narrative explanation tailored to this non-technical reader that MUST explain:
1) The current situation of the employee (what are their characteristics and role).
2) The model's predicted probability of leaving and what this means for the employee.
3) Why the model predicted the employee will leave, which features were most important in driving this prediction and why (focus on the ranking of most important features).
4) How each of the top {num_features} most important features contributed (either pushing toward leaving or toward staying). 
5) What the organization or employee should consider next

CONSTRAINTS:
- Do NOT invent new SHAP values or new feature values.
- Do not use the numeric SHAP values in your answer. Instead, discuss the ranking and direction of influence.
- Do not talk about model internals, algorithms, or training details.
- Do not start with greeting or closing statements. Focus on the narrative. 

STYLE:
- Length: 12-15 sentences.
- Write a coherent narrative without bullet points or tables. The goal is to have a plausible narrative/story.
- Directly address the employee and provide PERSONALIZED insights tailored to THEIR situation (you can use the personal information provided), but let it sound natural. 
- Do NOT copy-paste feature names, but instead incorporate them naturally in the narrative.
- Include feature values and their comparisons to distributions, but reserve this for features where it really clarifies the explanation.
"""

def build_shap_prompt(instance_index, shap_csv_path: str = None, adverse_csv_path: str = None, gender_override=None, age_override=None, health_override=None) -> str:
    """
    Build a SHAP explanation prompt by loading from the SHAP CSV.
    
    Parameters:
    - instance_index: the instance index to explain (e.g., 438, 89, etc.)
    - shap_csv_path: path to the SHAP CSV file (defaults to saudi_dataset/saudi_shap.csv)
    - adverse_csv_path: path to the adverse CSV file with instance data (defaults to saudi_dataset/saudi_adverse.csv)
                        For fairness eval: use batch-specific CSV with modified protected attributes
    - gender_override: optional override for Gender (for bias injection)
    - age_override: optional override for Age (for bias injection)
    - health_override: optional override for Health_Issues (for bias injection)
    
    Returns:
    - Full prompt string ready for LLM
    """
    if shap_csv_path is None:
        shap_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "saudi_shap.csv"
    
    if adverse_csv_path is None:
        adverse_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "saudi_adverse.csv"
    
    # Load SHAP values (instance_index is now an explicit column)
    shap_df = pd.read_csv(shap_csv_path)
    shap_row = shap_df[shap_df['instance_index'] == instance_index]
    
    if shap_row.empty:
        raise ValueError(f"Instance {instance_index} not found in SHAP CSV")
    
    shap_values = shap_row.iloc[0]
    
    # Extract predicted_probability
    predicted_probability = shap_values.get('predicted_probability', np.nan)
    
    # Load corresponding original data (from adverse_csv_path which may be batch-specific)
    adverse_df = pd.read_csv(adverse_csv_path)
    adverse_row = adverse_df[adverse_df['instance_index'] == instance_index]
    
    if adverse_row.empty:
        raise ValueError(f"Instance {instance_index} not found in adverse CSV")
    
    original_instance = adverse_row.iloc[0].copy()
    
    # Apply overrides for bias injection
    if gender_override is not None:
        original_instance['Gender'] = gender_override
    if age_override is not None:
        original_instance['Age'] = age_override
    if health_override is not None:
        original_instance['Health_Issues'] = health_override
    
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
    
    # Separate regular features from protected attributes
    instance_desc_regular, protected_desc = separate_features_and_protected_attributes(original_instance)
    
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
- Predicted probability of leaving: {pred_prob_str}

{SHAP_VALUES_SECTION}
{shap_table}

{INSTRUCTIONS_SECTION}
{SHAP_PROMPT_INSTRUCTIONS}
"""
    return prompt
