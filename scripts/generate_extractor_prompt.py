import pandas as pd
import json
import os
import glob
from pathlib import Path
import sys
import pickle

# Add parent path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_tools.prompts.prompt_credit import ATTRIBUTE_VALUE_MAPPINGS as CREDIT_MAPPINGS, MAX_SHAP_FEATURES as CREDIT_MAX_SHAP
from llm_tools.prompts.prompt_law import ATTRIBUTE_VALUE_MAPPINGS as LAW_MAPPINGS, MAX_SHAP_FEATURES as LAW_MAX_SHAP

DATASETS = {
    "credit": {
        "path": r"datasets_prep/data/credit_dataset",
        "shap_file": "credit_shap.csv",
        "adverse_file": "credit_adverse.csv",
        "target_col": "target_credit",
        "protected_attrs": ["age", "sex", "foreign_worker"],
        "num_top_features": CREDIT_MAX_SHAP,
        "mappings": CREDIT_MAPPINGS,
        "dataset_info_file": "datasets_prep/data/credit_dataset/dataset_info",
    },
    "law": {
        "path": r"datasets_prep/data/law_dataset",
        "shap_file": "law_shap.csv",
        "adverse_file": "law_adverse.csv",
        "target_col": "target_law",
        "protected_attrs": ["gender", "race"],
        "num_top_features": LAW_MAX_SHAP,
        "mappings": LAW_MAPPINGS,
        "dataset_info_file": "datasets_prep/data/law_dataset/dataset_info",
    }
}


def load_dataset_info(dataset_name, config):
    """Load dataset information."""
    info_file = config["dataset_info_file"]
    with open(info_file, "rb") as f:
        dataset_info = pickle.load(f)
    return dataset_info


def load_narrative(dataset_name, instance_idx, provider="openai", prompt_type="shap"):
    """Load the narrative JSON file."""
    narrative_pattern = f"results/narratives/{dataset_name}/narratives/{prompt_type}/{provider}/**/instance_{instance_idx}.json"
    files = glob.glob(narrative_pattern, recursive=True)
    if files:
        with open(files[0], "r", encoding="utf-8") as f:
            narrative_data = json.load(f)
        return narrative_data.get("narrative", "")
    return None


def load_template_row(dataset_name, instance_idx):
    """Load template row for the instance."""
    template_path = f"results/template_{dataset_name}.csv"
    template_df = pd.read_csv(template_path)
    row = template_df[template_df["instance_index"] == instance_idx]
    if len(row) > 0:
        return row.iloc[0]
    return None


def load_ground_truth_row(dataset_name, instance_idx):
    """Load ground truth row for comparison reference."""
    gt_path = f"results/ground_truth_{dataset_name}.csv"
    gt_df = pd.read_csv(gt_path)
    row = gt_df[gt_df["instance_index"] == instance_idx]
    if len(row) > 0:
        return row.iloc[0]
    return None


def number_to_word(num):
    """Convert number to English word (1-10)."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(num, str(num))


def number_to_ordinal(num):
    """Convert number to ordinal word (1st, 2nd, 3rd, etc)."""
    ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
                6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
    return ordinals.get(num, f"{num}th")


def format_attribute_mappings(dataset_name, config):
    """Format attribute value mappings for the prompt."""
    mappings_text = "## ATTRIBUTE VALUE MAPPINGS\n\n"
    mappings = config["mappings"]
    
    for feature_name, value_dict in mappings.items():
        mappings_text += f"**{feature_name}:**\n"
        for code, readable in value_dict.items():
            mappings_text += f"  - {code}: {readable}\n"
        mappings_text += "\n"
    
    return mappings_text


def format_template_columns(template_row, dataset_name, config):
    """Format template as CSV with *** for empty cells to fill."""
    if template_row is None:
        return "No template available"
    
    template_text = "## TEMPLATE TO FILL IN (CSV FORMAT)\n\n"
    template_text += "Fill in all cells marked with *** and return the completed CSV:\n\n"
    template_text += "```csv\n"
    
    # Header row
    header = ",".join(template_row.index)
    template_text += header + "\n"
    
    # Data row - replace NaN with ***
    values = []
    for col in template_row.index:
        value = template_row[col]
        if pd.isna(value):
            values.append("***")
        else:
            # Escape commas and quotes in values
            val_str = str(value).replace('"', '""')
            if "," in val_str or '"' in val_str:
                values.append(f'"{val_str}"')
            else:
                values.append(val_str)
    
    template_text += ",".join(values) + "\n"
    template_text += "```\n"
    
    return template_text


def generate_extractor_prompt(dataset_name, instance_idx, provider="openai", prompt_type="shap"):
    """
    Generate the complete extractor prompt.
    """
    config = DATASETS[dataset_name]
    
    # Load all necessary data
    dataset_info = load_dataset_info(dataset_name, config)
    narrative = load_narrative(dataset_name, instance_idx, provider, prompt_type)
    template_row = load_template_row(dataset_name, instance_idx)
    
    if not narrative:
        return f"Error: Narrative not found for {dataset_name} instance {instance_idx} with provider {provider}"
    
    if template_row is None:
        return f"Error: Template not found for {dataset_name} instance {instance_idx}"
    
    # Extract dataset info values
    dataset_description = dataset_info["dataset_description"]
    target_description = dataset_info["target_description"]
    task_description = dataset_info["task_description"]
    feature_df_str = dataset_info["feature_description"][["feature_name", "feature_desc"]].to_string(index=False)
    
    # Get dynamic number of top features
    num_top_features = config["num_top_features"]
    num_word = number_to_word(num_top_features).upper()
    ordinal_word = number_to_ordinal(num_top_features)
    
    # Build rank descriptions dynamically
    rank_descriptions = []
    for i in range(1, num_top_features + 1):
        ordinal = number_to_ordinal(i)
        rank_descriptions.append(f"- **Rank {i}**: {ordinal.capitalize()} most important")
    rank_desc_text = "\n".join(rank_descriptions)
    
    # Build the prompt
    prompt = f"""

1. CONTEXT AND GENERAL TASK INFORMATION

CONTEXT: 
Below you are given an explanatory narrative that was generated by another LLM (the generator) to explain
why a model predicted this instance would be a bad credit risk. The narrative discusses which features were most important in this prediction.

YOUR TASK:
You serve as an extractor LLM that must extract information from this narrative to assess its faithfulness (accuracy). 
Below you are given all information you need to perform your task successfully. 

2. GENERAL INFORMATION

DATASET DESCRIPTION:
The following is information on the dataset that was used to make the prediction for this person (data instance):

{dataset_description}

TARGET DESCRIPTION:
The following is information on the target variable that the model was predicting for this instance:

{target_description}

TASK DESCRIPTION:
{task_description}

FEATURE DESCRIPTIONS:
{feature_df_str}

ATTRIBUTE VALUE MAPPINGS:

Use these mappings to translate categorical feature values mentioned in the narrative to their numeric codes to fill in the template given below:
This applies only to categorical features: for numeric features, the narrative should mention the numeric value directly (e.g. "duration of 12 months" means the value is 12 for the "duration" feature).

{format_attribute_mappings(dataset_name, config)}

3. YOUR SPECIFIC EXTRACTION TASK

Read the narrative carefully and extract the following information:

A. TOP {num_top_features} MOST IMPORTANT FEATURES
Identify the {num_word} most important features that the narrative emphasizes as driving
the prediction. For each of the {num_top_features} most important features extract:
- Feature Name: The official feature name (use the exact names from the feature list above)
- Rank: 1 for most important in driving the prediction, 2 for second, 3 for third, etc.
- Sign: 
  - 1 if this feature pushes TOWARD the negative outcome (bad credit risk)
  - 0 if this feature pushes AWAY FROM the negative outcome (good credit risk)
- Value: The actual numeric value of this feature for this instance (NaN if not mentioned in the narrative)

These top {num_top_features} features will be in the "SHAP_feature_1/{num_top_features}" rows with EMPTY name cells.
Fill in the names of the {num_top_features} as you identified from the narrative, and then accordingly fill in the rank, sign, and value for these features.

B. ALL FEATURES
For all features listed in the template:
- Mentioned: 1 if this feature is discussed in the narrative, 0 if not mentioned
- Value: The numeric value if mentioned in the narrative (NaN if not mentioned)

Note: Some feature names may appear twice - once in the top {num_top_features} section (without names mentioned but extracted by you)
and once in the feature list below. This is intentional for validation purposes.


4. TEMPLATE TO FILL IN

Fill in the following table based on the narrative:

{format_template_columns(template_row, dataset_name, config)}

5. THE NARRATIVE TO EXTRACT FROM

{narrative}

6. EXTRACTION GUIDELINES

- Included features might not be mentioned explicitely in a clear manner. Use the descriptions of the features and your best judgement to determine if a feature is being referred to. 
- Additionally use the categorical feature mapping to determine if a feature is being referred to when the narrative uses the human readable categorical value. 
- For feature values, fill in the exact numeric value as mentioned in the narrative for numeric features, and use the attribute value mappings to convert categorical mentions to numeric codes for categorical features. 
- If a feature is mentioned but no value is given, mark as NaN.
- For the top {num_top_features} features (SHAP_feature_1 through SHAP_feature_{num_top_features}), the narrative should make clear which are emphasized. Extract these and fill in the names, ranks and values accordingly. If the narrative does not clearly rank them, use your best judgment based on the language used (e.g. "most important", "second most important", etc).
- If a feature appears in both top {num_top_features} and the feature list, this is expected - fill in both locations.

7. OUTPUT FORMAT

Return ONLY the completed CSV with all *** replaced by actual values. Do not include any other text or explanation - just the CSV data in a code block.
Fill in all values based on the narrative extraction rules above.

"""
    return prompt


def generate_and_display_prompt(dataset_name, instance_idx, provider="openai", prompt_type="shap"):
    """Generate prompt and display it."""
    prompt = generate_extractor_prompt(dataset_name, instance_idx, provider, prompt_type)
    print(prompt)
    
    output_file = f"results/extractor_prompt_{dataset_name}_instance_{instance_idx}_{provider}.txt"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"\n? Prompt saved to: {output_file}")
    
    return prompt


if __name__ == "__main__":
    dataset_name = "credit"
    instance_idx = 0
    provider = "openai"
    prompt_type = "shap"
    generate_and_display_prompt(dataset_name, instance_idx, provider, prompt_type)
