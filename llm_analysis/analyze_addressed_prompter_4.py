import json
import pandas as pd
from config import *

task_description_for_analyze_addressed_7 = \
                   "Input Format:\n" \
                   "Code Review:\n" \
                   "- Reviewed Path: The file path of the code change.\n" \
                   "- Reviewed Change: The code change under review. (\"+\" indicates added lines, \"-\" indicates removed lines)\n" \
                   "- Review Comment: A list of issues or suggestions for the above changes.\n" \
                   "Subsequent Code Changes: " \
                   "A cumulative code changes made to the reviewed file after the review." \
                   "\n\n" \
                   "Task: For each item in the Review Comment, " \
                   "analyze whether it has been addressed by the Subsequent Code Changes by determining if:\n" \
                   "- The issue has been completely resolved.\n" \
                   "- The suggestion has been completely followed. " \
                   "(If the suggestion includes concrete code examples, it does not need to be followed exactly. "\
                   "As long as the code change serves the same intended purpose, the suggestion can be considered addressed.)\n" \
                   "After completing the analysis, classify the entire review comment into one of the following categories:\n" \
                   "- Not Addressed: None of the issues or suggestions have been addressed by the Subsequent Code Changes.\n" \
                   "- Partly Addressed: Some but not all issues or suggestions have been addressed.\n" \
                   "- Fully Addressed: All issues or suggestions have been addressed.\n" \
                   "If the code relevant to the Review Comment has been completely removed in the Subsequent Code Changes, " \
                   "and it is not possible to determine whether the removal was an intentional response to the comment, " \
                   "you can classify it as \"Not Enough Information\"." \
                   "\n\n" \
                   "Output Format:\n" \
                   "Analysis details:\n" \
                   "- Addressed Items: List the issues or suggestions that have been addressed.\n" \
                   "- Unaddressed Items: List the issues or suggestions that have not been addressed.\n"\
                   "Classification: " \
                   "Based on the Addressed Items and Unaddressed Items, " \
                   "select one of \"Not Addressed\", \"Partly Addressed\", \"Fully Addressed\", or \"Not Enough Information\" " \
                   "as the final classification without more explanations."

task_description_for_analyze_addressed = task_description_for_analyze_addressed_7

def describe_review(file_name,review_change,review_content,suggestion_items):
    path = json.dumps(file_name, ensure_ascii=False)
    diff_hunk = json.dumps(review_change, ensure_ascii=False)
    comment = json.dumps(review_content, ensure_ascii=False)
    items = json.dumps(suggestion_items, ensure_ascii=False)
    review_description = "Code Review:\n" \
                        f"- Reviewed Path: {path}\n" \
                        f"- Reviewed Change: {diff_hunk}\n" \
                        f"- Review Comment: {items}"
    return review_description

#File renamed without changes.
def describe_code_change(code_change,spec_filename,new_file_name):
    spec_filename = json.dumps(spec_filename, ensure_ascii=False)
    code_change = json.dumps(code_change, ensure_ascii=False)

    commits_description = ""
    # Case 1: The file was deleted (new_file_name is empty and no further modifications were made).
    if code_change=="\"File_Deleted\"" :
        commits_description = f"Subsequent Code Changes: The reviewed {spec_filename} file has been deleted after the Code Review."
    if new_file_name:
        new_file_name = json.dumps(new_file_name, ensure_ascii=False)
        # Case 2: The file was renamed with no content modifications.
        if code_change=="":
            commits_description =  f"Subsequent Code Changes: The reviewed {spec_filename} file has been renamed as {new_file_name} after the Code Review. " \
                                   f"The content of the file remains unchanged."
        # Case 3: The file was renamed and its content was modified.
        else:
            commits_description = f"Subsequent Code Changes: The reviewed {spec_filename} file has been renamed as {new_file_name}. " \
                                  f"Here are the cumulative modifications made to the renamed file {new_file_name} after the Code Review.\n"
            commits_description += code_change
    # Case 4: The file content was modified.
    if len(commits_description)==0:
        commits_description = f"Subsequent Code Changes: Here are the cumulative modifications made to the reviewed file {spec_filename} after the Code Review.\n"
        commits_description += code_change

    return commits_description.strip()


def constract_prompt_for_analyse_addressed(file_name,new_file_name,reviewed_change,review_content,suggestion_items,subsequent_change):
    review_description = describe_review(file_name,reviewed_change,review_content,suggestion_items)
    commits_description = describe_code_change(subsequent_change, file_name, new_file_name)
    prompt = task_description_for_analyze_addressed \
             + "\n\nInput:\n" \
             + review_description \
             + "\n\n" \
             + commits_description \
             + "\n\nOutput: List the addressed and unaddressed items, and make the final classification.\n"
    return prompt

