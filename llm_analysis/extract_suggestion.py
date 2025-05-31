import csv
import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from call_llm import get_model_function
from config import BASE_DIR
from llm_analysis.extract_suggestion_prompter_3 import constract_prompt_for_indentify_suggestion as constract_prompt_for_indentify_suggestion_3


llm_model = "openai-gpt-4o"
llm_model = "claude-3-5-haiku"
llm_model = "deepseek-v3"
llm_model = "deepseek-r1"
llm_model = "openai-o3-mini"
llm_model = "openai-gpt-4o-mini"
llm_model = "claude-3-7-sonnet"
llm_model = "openai-gpt-4.1"
response_function = get_model_function(llm_model, input_type="prompt")

prompt_function = constract_prompt_for_indentify_suggestion_3

def gpt_indentify_suggestions(file_name,change_content,review_content):
    prompt = prompt_function(file_name,change_content,review_content)
    response = response_function(prompt)
    return prompt, response

def indentify_suggestion_in_comment(review_file,
                                    output_file, output_record_file):

    # Write run information to the record file
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "review_file": review_file,
        "output_file": output_file,
        "llm_model": llm_model
    }
    with open(output_record_file, 'a', encoding='utf-8') as record_file:
        record_file.write(json.dumps(run_info) + "\n\n")

    # Load data:
    review_details = pd.read_csv(review_file)

    # Check whether Comment_URL is present; if not, use Comment_ID instead.
    if 'Comment_URL' not in review_details.columns and 'Comment_ID' in review_details.columns:
        review_details = review_details.rename(columns={'Comment_ID': 'Comment_URL'})

    # Prepare the output file:
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['Comment_URL', 'GPT_Input' ,'GPT_Output'])
            writer.writeheader()

    # Check which data has already been analyzed:
    existing_results = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_results.add(row['Comment_URL'])

    # Process each review
    for _, row in review_details.iterrows():
        # Skip the parts that have already been analyzed:
        if row['Comment_URL'] in existing_results:
            continue;
        result = indentify_for_single_review(row)
        if result is not None:
            with open(output_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['Review_URL', 'GPT_Input', 'GPT_Output'])
                writer.writerow(result)

    print(f"Analysis results incrementally saved to {output_file}")

def indentify_for_single_review(row):

    review_url = row['Comment_URL']
    print(f"Processing for {review_url} with {llm_model}")

    file_name = row["Diff_path"]
    change_content = row["Diff_hunk"]
    review_content = row["Body"]

    # Whether the comment contains issues or suggestions.
    prompt, resolution = gpt_indentify_suggestions(file_name,change_content,review_content)
    # print("============================= Prompt =============================")
    # print(prompt)
    # print("============================= Resolution =============================")
    # print(resolution)

    result = {
        'Review_URL': review_url,
        'GPT_Input': prompt,
        'GPT_Output': resolution
    }

    return result


def format_suggestion(input_file, output_file, classification_type=2):
    # Read input file
    data = pd.read_csv(input_file)

    # Apply the parsing function to the 'GPT_Output' column
    data[['Classification', 'Issue_Suggestion_List']] = data['GPT_Output'].apply(
        lambda x: pd.Series(parse_suggestion(x, classification_type))
    )

    # Keep only required columns
    formatted_data = data[['Comment_URL', 'Classification', 'Issue_Suggestion_List']]

    # Save to output file
    formatted_data.to_csv(output_file, index=False)
    print(f"Formatted data has been saved to {output_file}")


# Constants
TAG_FOR_LIST = "issues or suggestions:"
TAG_FOR_CLASSIFICATION = "classification:"

# Classification mappings
CLASSIFICATION_MAPPINGS = {
    # Binary classification (format_suggestion_1)
    1: {
        "mapping": {
            "no issue or suggestion": 0,
            "contains issues or suggestions": 1
        },
        "no_list_needed": [0]
    },
    # Three-way classification (format_suggestion_2)
    2: {
        "mapping": {
            "not contain any issues or suggestions": 0,
            "only contain general issues or suggestions": 1,
            "contain valid issues or suggestions": 2
        },
        "no_list_needed": [0, 1]
    }
}
def parse_suggestion(suggestion_text, classification_type):
    list_content = ""
    classification = -1
    try:
        suggestion_text = suggestion_text.lower()

        classification_start_index = suggestion_text.index(TAG_FOR_CLASSIFICATION) + len(TAG_FOR_CLASSIFICATION)
        classification_content = suggestion_text[classification_start_index:].strip()

        # Determine classification value using the mapping
        classification_mapping = CLASSIFICATION_MAPPINGS.get(classification_type, {})
        mapping = classification_mapping.get("mapping", {})
        for key, value in mapping.items():
            if key in classification_content:
                classification = value
                break

         # Extract list content and classification content
        # Check if list extraction is needed
        no_list_needed = classification_mapping.get("no_list_needed", [])
        if classification not in no_list_needed:
            try:
                list_start_index = suggestion_text.index(TAG_FOR_LIST) + len(TAG_FOR_LIST)
                list_content = suggestion_text[
                               list_start_index:classification_start_index - len(TAG_FOR_CLASSIFICATION)
                               ].strip()
            except ValueError:
                list_content = ""  # Gracefully skip if list section not found

    except Exception as e:
        print(f"Error parsing suggestion: {e}")

    if classification == -1:
        print(f"Error parsing suggestion\n: {suggestion_text}")

    return classification, list_content

