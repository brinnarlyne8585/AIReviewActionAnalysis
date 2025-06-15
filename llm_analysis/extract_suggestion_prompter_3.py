import json
import pandas as pd
from config import *

task_description_for_indentify_suggestion_12 = \
                   "Input Format:\n" \
                   "Code Review:\n" \
                   "- Reviewed Path: The file path of the code change.\n" \
                   "- Reviewed Change: The code change under review. (\"+\" indicates added lines, \"-\" indicates removed lines)\n" \
                   "- Review Comment: A comment for the above changes, which may includes issues or suggestions." \
                   "\n\n" \
                   "Task: " \
                   "List the issues or suggestions raised in the Review Comment, and determine whether they are valid or not.\n" \
                   "The valid issues or suggestions should be:\n" \
                   "- Focus on the reviewed file itself.\n" \
                   "- Actionable: Require a change to the reviewed file.\n" \
                   "- Specific: Includes details to locate which parts of the Reviewed Change should be modified.\n" \
                   "(Suggestions that are expressed in a indirect manner are acceptable, including those framed as questions.)\n" \
                   "The following are considered only general issues or suggestions, but NOT valid:\n" \
                   "- Comments that lack any specific details, and could apply to any code review scenario.\n"\
                   "- Comments that only request checking of resources, without requiring specific code modifications.\n" \
                   "The following are NOT considered issues or suggestions at all:\n" \
                   "- Items that have been implemented in the Reviewed Change, as they are the descriptions of the existing code changes.\n" \
                   "- Suggestions that are unrelated to the reviewed file (e.g., requests for more context).\n" \
                   "After completing the analysis, classify the entire review comment into one of the following categories:\n" \
                   "- Not Contain Any Issues Or Suggestions: The review comment contains no issues or suggestions.\n" \
                   "- Only Contain General Issues Or Suggestions: The review comment only contains general issues or suggestions.\n" \
                   "- Contain Valid Issues Or Suggestions: The review comment contains at least one valid issues or suggestions." \
                   "\n\n" \
                   "Output Format:\n" \
                   "Issues or Suggestions: List valid items from the Review Comment clearly, " \
                   "preserving code details, including inline code blocks and multi-line code snippets, " \
                   "and using the original wording as much as possible.\n" \
                   "Classification: " \
                   "Based on the listed items, " \
                   "select one of \"Not Contain Any Issues Or Suggestions\", \"Only Contain General Issues Or Suggestions\" or \"Contain Valid Issues Or Suggestions\" " \
                   "as the final classification without more explanations."

task_description_for_indentify_suggestion = task_description_for_indentify_suggestion_12

def describe_review(file_name,change_content,review_content):
    diff_hunk = json.dumps(change_content, ensure_ascii=False)
    path = json.dumps(file_name, ensure_ascii=False)
    comment = json.dumps(review_content, ensure_ascii=False)
    review_description = "Code Review:\n" \
                         f"- Reviewed Path: {path}\n" \
                         f"- Reviewed Change: {diff_hunk}\n" \
                         f"- Review Comment: {comment}"
    return review_description


def constract_prompt_for_indentify_suggestion(file_name,change_content,review_content):
    review_description = describe_review(file_name,change_content,review_content)
    prompt = task_description_for_indentify_suggestion \
             + "\n\nInput:\n" \
             + review_description \
             + "\n\nOutput: List the identified issues or suggestions, and make the final classification.\n"
    return prompt


