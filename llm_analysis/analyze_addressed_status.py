import csv
import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from call_llm import get_model_function
from config import BASE_DIR
from llm_analysis.analyze_addressed_prompter_4 import constract_prompt_for_analyse_addressed as constract_prompt_for_analyse_addressed_4

######################################## Subtask 2: Extract issues or suggestions from the reviews ########################################

llm_model = "openai-gpt-4o"
llm_model = "openai-gpt-4.1"
llm_model = "deepseek-v3"
llm_model = "claude-3-5-haiku"
llm_model = "openai-gpt-4o-mini"
llm_model = "claude-3-7-sonnet"
llm_model = "deepseek-r1"
llm_model = "openai-gpt-4.1"
llm_model = "openai-o3-mini"
response_function = get_model_function(llm_model, input_type="prompt")

prompt_function = constract_prompt_for_analyse_addressed_4

def gpt_analyze_addressed_state(file_name,new_file_name,reviewed_change,review_content,suggestion_items,subsequent_change):
    prompt = prompt_function(file_name,new_file_name,reviewed_change,review_content,suggestion_items,subsequent_change)
    response = response_function(prompt)
    return prompt, response


csv.field_size_limit(sys.maxsize)
def analyze_review_abbressed_state(review_file, suggestion_items_file,
                                   output_file, output_record_file):
    # Write run information to the record file
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "review_file": review_file,
        "suggestion_items_file": suggestion_items_file,
        "output_file": output_file,
    }
    with open(output_record_file, 'a', encoding='utf-8') as record_file:
        record_file.write(json.dumps(run_info) + "\n\n")

    # Load data:
    review_file = pd.read_csv(review_file)

    # Check if Comment_URL is present; if not, replace it with Comment_ID.
    if 'Comment_URL' not in review_file.columns and 'Comment_ID' in review_file.columns:
        review_file = review_file.rename(columns={'Comment_ID': 'Comment_URL'})

    suggestion_items = pd.read_csv(suggestion_items_file)

    # Append ChatGPT's analysis results to the end of review_details.
    review_file = review_file.merge(
        suggestion_items.rename(
            columns={"Classification": "Is_Contain_Item", "Issue_Suggestion_List": "Issue_Suggestion_List"}),
        on="Comment_URL",
        how="left"
    )

    # Prepare the output file:
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['Comment_URL', 'GPT_Input', 'GPT_Output'])
            writer.writeheader()

    # Load previously processed data to avoid redundant processing.
    existing_results = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_results.add(row['Comment_URL'])

    # Iterate over the relation_file data for processing.
    for _, row in review_file.iterrows():

        # Skip the already analyzed parts:
        if row['Comment_URL'] in existing_results:
            continue;

        # Process review
        result = analyze_single_review(row)

        if result is not None:
            # Save the results.
            with open(output_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['Comment_URL', 'GPT_Input', 'GPT_Output'])
                writer.writerow(result)

    print(f"Analysis results incrementally saved to {output_file}")

not_any_item_message = "Not contain any items."
no_specific_item_message = "Not contain specific items."
no_change_message = "No change happened; the comment remains unaddressed."
def analyze_single_review(review_row):

    review_url = review_row['Comment_URL']

    print(f"Processing for {review_url} with {llm_model}")

    # Analyze whether the review comment contains issues or suggestions.
    if review_row['Is_Contain_Item']==0:
        message = not_any_item_message
        prompt, resolution = message, message
    elif review_row['Is_Contain_Item']==1:
        message = no_specific_item_message
        prompt, resolution = message, message
    elif (pd.isna(review_row['New_path']) or review_row['New_path'] == "") and \
            (pd.isna(review_row['Change_Until_Merged']) or review_row['Change_Until_Merged'] == ""):
        message = no_change_message
        prompt, resolution = message, message
    else:
        file_name = review_row["Diff_path"]
        new_file_name = "" if pd.isna(review_row.get("New_path")) else review_row.get("New_path")
        review_change = review_row["Diff_hunk"]
        review_content = review_row["Body"]
        suggestion_items = review_row["Issue_Suggestion_List"]
        subsequent_change = "" if pd.isna(review_row.get("Change_Until_Merged")) else review_row.get("Change_Until_Merged")
        prompt, resolution = gpt_analyze_addressed_state(file_name,new_file_name,review_change,review_content,suggestion_items,subsequent_change)

    # print("============================= Prompt =============================")
    # print(prompt)
    # print("============================= Resolution =============================")
    # print(resolution)

    result = {
        'Comment_URL': review_row["Comment_URL"],
        'GPT_Input': prompt,
        'GPT_Output': resolution
    }

    return result


def format_abbressed_state(input_file, output_file):
    """
    Convert GPT analysis results into a formatted file based on the content of the Resolution field.

    :param input_file: Path to the input file (e.g., gpt_output_1.csv)
    :param output_file: Path to the output file (e.g., gpt_format_1.csv)
    """
    # Read the input file.
    data = pd.read_csv(input_file)

    def get_index(text, sub_text, start_index):
        return text.index(sub_text, start_index) if sub_text in text[start_index:] else len(text)

    # Define the parsing method.
    def resolve_format(resolution, review_url):

        format_result = -2
        if resolution.strip() == not_any_item_message:
            format_result = "0"

        if resolution.strip() == no_specific_item_message:
            format_result = "1"

        if resolution.strip() == no_change_message:
            format_result = "2,0"

        if "classification" in resolution.lower():
            resolution = resolution.lower()
            start_index = resolution.index("classification")
            type_minus_one_index = get_index(resolution, "not enough information", start_index)
            type_two_index = get_index(resolution, "not addressed", start_index)
            type_three_index = get_index(resolution, "partly addressed", start_index)
            type_four_index = get_index(resolution, "fully addressed", start_index)
            final_index = min(type_minus_one_index,type_two_index,type_three_index,type_four_index)
            if final_index==type_minus_one_index:
                format_result = "2,-1"
            if final_index==type_two_index:
                format_result = "2,0"
            elif final_index==type_three_index:
                format_result = "2,1"
            elif final_index==type_four_index:
                format_result = "2,2"

        if format_result==-2:
            print(f"[WARN] Failed to parse Resolution for Review_URL: {review_url} — Resolution: {resolution}")

        return format_result

    # Create a new column named Resolution_Formatted.
    data['Resolution_Formated'] = data.apply(
        lambda row: resolve_format(row['GPT_Output'], row['Comment_URL']), axis=1
    )

    # Keep only the required columns.
    formatted_data = data[['Comment_URL', 'Resolution_Formated']]

    # Save to the output file.
    formatted_data.to_csv(output_file, index=False)
    print(f"Formatted data has been saved to {output_file}")
