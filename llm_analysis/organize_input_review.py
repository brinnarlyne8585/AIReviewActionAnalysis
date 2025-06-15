import os
import re
import pandas as pd
import difflib
from crawlers import crawler_infrastructure as crawler
from config import BASE_DIR

# action_has_human_review
action_with_human_review = [
    "anc95/ChatGPT-CodeReview",
    "mattzcarey/code-review-gpt",
    "coderabbitai/ai-pr-reviewer",
]

# patch_level review action
patch_level_review_action = [
    "coderabbitai/ai-pr-reviewer",
    "aidar-freeed/ai-codereviewer",
]

file_level_review_action = [
    "anc95/ChatGPT-CodeReview",  # 1
    "mattzcarey/code-review-gpt",  # 2
]

######################## Structuring data to help a large language model determine whether a comment has been addressed.  ########################
File_Deleted = "File_Deleted"
def organize_input_review(review_file, file_version_source, output_file, review_level="patch"):

    review_df = pd.read_csv(review_file)

    # Read the file version from the final merged commit.
    if os.path.exists(file_version_source):
        file_source = pd.read_parquet(file_version_source)
    else:
        print(f"[ERROR] File version source not found: {file_version_source}")
        return

    # Used to store the reshaped data.
    reshaped_data = []

    # Iterate over each review comment.
    for idx, row in review_df.iterrows():
        pr_url = row["Affiliated_PR_URL"]
        repo_info = crawler.extract_owner_repo(pr_url)
        comment_url = row["Comment_URL"]

        review_content = row["Body"]
        diff_path = row["Diff_path"]

        original_commit_id = row["Original_Commit_id"]

        # Retrieve comments with processed suggested code blocks.
        review_content = shape_review_comment(row)

        # Obtain the review's change perspective.
        if review_level=="patch":
            reviewed_change = shape_change_for_patch_review(row)
        elif review_level=="file" and "Full_Diff_Hunk" in row:
            reviewed_change = row["Full_Diff_Hunk"]
        elif review_level=="file" and "Diff_hunk" in row:
            reviewed_change = row["Diff_hunk"]

        # Retrieve the changes from the review version to the merged version.
        merge_commit_sha = row["Merge_Commit_SHA"]
        code_diff = get_code_diff(file_source, repo_info, diff_path, original_commit_id, merge_commit_sha)

        new_diff_path = ""
        merged_file_row = file_source[
                (file_source["Repo_Info"] == repo_info) &
                (file_source["Diff_Path"] == diff_path) &
                (file_source["Commit_id"] == merge_commit_sha)
                ]
        if not merged_file_row.empty:
            new_diff_path = merged_file_row.iloc[0]["New_Diff_Path"]

        reshaped_data.append({
            "Comment_URL": comment_url,
            "Review_Start_Line": row["Original_Start_Line"] if "Original_Start_Line" in row and pd.notna(row["Original_Start_Line"]) else "",
            "Review_End_Line": row["Original_Line"] if "Original_Line" in row and pd.notna(row["Original_Line"]) else "",
            "Original_Commit_id": original_commit_id,
            "Merge_Commit_id": merge_commit_sha,
            "Diff_path": diff_path,
            "New_path": new_diff_path,
            "Body": review_content,
            "Diff_hunk": reviewed_change,
            "Change_Until_Merged": code_diff
        })

    reshaped_df = pd.DataFrame(reshaped_data)

    # Save results
    if os.path.exists(output_file):
        os.remove(output_file)
    reshaped_df.to_csv(output_file, index=False)

    print(f"[INFO] Analysis completed. Results saved to {output_file}")


def shape_review_comment(row):
    ori_comment = row["Body"]
    full_block,suggestion_block = extract_suggestion_block(ori_comment)
    if full_block==None:
        return ori_comment
    old_code = shape_change_for_patch_review(row, window_size=0)
    new_code = "+"+suggestion_block.replace("\n","\n+")
    block_translation = f"I suggest changing\n ```\n{old_code}\n```\n to\n```\n{new_code}\n```"
    ori_comment = ori_comment.replace(full_block,block_translation)
    return ori_comment


def extract_suggestion_block(text):
    """
    Extract the contents of suggestion ... code blocks from the text.
    """
    pattern = r"(```suggestion\n([\s\S]*?)\n```)"
    match = re.search(pattern, text)
    if match:
        full_block = match.group(1)  # The complete suggestion ... block.
        extracted_content = match.group(2)  # The content inside the code block.
        return full_block, extracted_content
    return None, None

def get_review_line_num_for_multiple_line_review_2(diff_hunk_lines,original_start_line,original_line):
    start_index = min(original_start_line, original_line)
    end_index = max(original_start_line, original_line)
    # Parse the header of the diff hunk.
    header = diff_hunk_lines[0]
    _, _, new_info, _ = header.split(" ")[0:4]
    new_start, _ = map(int, new_info[1:].split(","))
    # Compute the mapping of modified line numbers.
    current_line = new_start-1
    new_lines = []
    for line in diff_hunk_lines[1:]:
        if not line.startswith("-"):  # Count only the lines in the new file.
            current_line += 1
        new_lines.append((current_line, line))
    # Count the number of lines between start_line and end_line.
    review_line_num = sum(1 for ln, _ in new_lines if start_index <= ln <= end_index)
    return review_line_num


def shape_change_for_patch_review(row, window_size=3):
    diff_hunk = row["Diff_hunk"]
    diff_hunk_lines = diff_hunk.splitlines()
    # If Original_Start_Line is empty, it indicates a single-line review: take the last four lines of the diff_hunk as change_for_review.
    if pd.isna(row["Original_Start_Line"]):
        review_line_num = 1+window_size
    else:
        original_start_line = int(row["Original_Start_Line"])
        original_line = int(row["Original_Line"])
        review_line_num = get_review_line_num_for_multiple_line_review_2(diff_hunk_lines,original_start_line,original_line)
    # Take the last four lines.
    last_four_lines = diff_hunk_lines[-review_line_num:]
    # Join using newline characters to form a new string.
    change_for_review = "\n".join(last_four_lines)
    return change_for_review


def get_file_content(file_source, repo_info, diff_path, commit_id):
    file_row = file_source[
        (file_source["Repo_Info"] == repo_info) &
        (file_source["Diff_Path"] == diff_path) &
        (file_source["Commit_id"] == commit_id)
    ]
    if file_row.empty:
        return None
    return remove_bom(file_row.iloc[0]["File_Content"])


def get_code_diff(file_source, repo_info, diff_path, original_commit_id, merge_commit_sha):
    """
    Retrieve the code changes.
    """
    # Retrieve the file content from the merged commit version.
    merged_content = get_file_content(file_source, repo_info, diff_path, merge_commit_sha)

    # Retrieve the file content from the original commit version.
    original_content = get_file_content(file_source, repo_info, diff_path, original_commit_id)

    if merged_content==None:
        if original_content==None:
            return ""
        else:
            return File_Deleted

    if original_content==None and merged_content!=None:
        print(f"[WARNING] No original commit file found for {diff_path} at {original_commit_id}")
        return None

    # Compare whether the code lines were modified in the merged commit and extract the change blocks.
    code_diff = compare_diff(original_content, merged_content)

    return code_diff

def remove_bom(s: str) -> str:
    """
    Remove the UTF-8 Byte Order Mark (BOM) from the beginning of the string.
    """
    BOM = '\ufeff'  # The Unicode representation of the UTF-8 Byte Order Mark (BOM) is U+FEFF.
    if s.startswith(BOM):
        return s[len(BOM):]
    return s

def compare_diff(original_content, merged_content):

    old_lines = original_content.splitlines()
    new_lines = merged_content.splitlines()

    diff_lines = difflib.unified_diff(old_lines,new_lines)
    diff_string = '\n'.join(diff_lines)

    return diff_string


def pipeline_for_human_review():

    for AI_reviewer in action_with_human_review:
        # print(f"[INFO] Current Processing for: {AI_reviewer}")
        AI_reviewer_refined_for_path = AI_reviewer.replace("/", "_")
        directory_for_AI_reviewer = f"{BASE_DIR}/_data/{AI_reviewer_refined_for_path}"
        crawling_directory = f"{directory_for_AI_reviewer}/crawled_data"
        llm_input_directory = f"{directory_for_AI_reviewer}/llm_input"

        if not os.path.exists(llm_input_directory):
            os.makedirs(llm_input_directory)

        ###### Structuring data to help a large language model determine whether a comment has been addressed. ######
        human_review_file = f"{crawling_directory}/valid_human_reviews.csv"
        file_version_source = f"{crawling_directory}/reviewed_file_versions.parquet"
        output_file = f"{llm_input_directory}/valid_human_reviews(llm_input)(consider_path).csv"
        organize_input_review(human_review_file, file_version_source, output_file)


def pipeline_for_patch_level_review():

    for AI_reviewer in patch_level_review_action:
        print(f"[INFO] Current Processing for: {AI_reviewer}")
        AI_reviewer_refined_for_path = AI_reviewer.replace("/", "_")
        directory_for_AI_reviewer = f"{BASE_DIR}/_data/{AI_reviewer_refined_for_path}"
        crawling_directory = f"{directory_for_AI_reviewer}/crawled_data"
        llm_input_directory = f"{directory_for_AI_reviewer}/llm_input"

        if not os.path.exists(llm_input_directory):
            os.makedirs(llm_input_directory)

        ###### Structuring data to help a large language model determine whether a comment has been addressed. ######
        review_file = f"{crawling_directory}/valid_reviews.csv"
        file_version_source = f"{crawling_directory}/reviewed_file_versions.parquet"
        output_file = f"{llm_input_directory}/valid_reviews(llm_input)(consider_path).csv"
        organize_input_review(review_file, file_version_source, output_file)


def pipeline_for_file_level_review():

    for AI_reviewer in file_level_review_action:
        # print(f"[INFO] Current Processing for: {AI_reviewer}")
        AI_reviewer_refined_for_path = AI_reviewer.replace("/", "_")
        directory_for_AI_reviewer = f"{BASE_DIR}/_data/{AI_reviewer_refined_for_path}"
        crawling_directory = f"{directory_for_AI_reviewer}/crawled_data"
        llm_input_directory = f"{directory_for_AI_reviewer}/llm_input"

        if not os.path.exists(llm_input_directory):
            os.makedirs(llm_input_directory)

        ###### Structuring data to help a large language model determine whether a comment has been addressed.  ######
        review_file = f"{crawling_directory}/valid_reviews(diff_reshaped).csv"
        file_version_source = f"{crawling_directory}/reviewed_file_versions.parquet"
        output_file = f"{llm_input_directory}/valid_reviews(diff_reshaped)(llm_input)(consider_path).csv"
        review_level = "file"
        organize_input_review(review_file, file_version_source, output_file, review_level)



if __name__ == "__main__":

    pipeline_for_human_review()
    pipeline_for_patch_level_review()
    pipeline_for_file_level_review()
