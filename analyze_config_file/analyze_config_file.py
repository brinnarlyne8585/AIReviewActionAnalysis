import csv
import datetime
import os
from typing import Set
import pandas as pd

from action_config_map import config_map_for_required, value_list_for_required
from config import BASE_DIR
from yaml_parse.parser import compare_config_content,focus_config_data,get_config_data


review_action = [
    "anc95/ChatGPT-CodeReview",  # 1
    "coderabbitai/ai-pr-reviewer",  # 3
    "aidar-freeed/ai-codereviewer",  # 4

    "truongnh1992/gemini-ai-code-reviewer",  # 9
    "tmokmss/bedrock-pr-reviewer",  # 12
    "presubmit/ai-reviewer",  # 14
    "gvasilei/AutoReviewer",  # 16

    "unsafecoerce/chatgpt-action",  # 17
    "magnificode-ltd/chatgpt-code-reviewer",  # 19
]

comment_action = [
    "mattzcarey/code-review-gpt",  #2
    "kxxt/chatgpt-action",  #5
    "cirolini/genai-code-review",  # 6
    "feiskyer/ChatGPT-Reviewer",  # 10
    "adshao/chatgpt-code-review-action",  # 11
    "Integral-Healthcare/robin-ai-reviewer",  # 13
    "ca-dp/code-butler",  # 20
]

all_action = []
all_action.extend(review_action)
all_action.extend(comment_action)


review_action_name = [x for x in all_action]
review_action_name.append("freeedcom/ai-codereviewer") # another name of aidar-freeed/ai-codereviewer
def get_review_action_name(file_content):
    for action in review_action_name:
        if action in file_content:
            return action;
    return None

def analyze_code_diff_stats(valid_repos, commit_history_file, output_file, do_fetch=False):
    """
    Analyze the crawled modification history and output statistical results, including:
    -Detailed information for each configuration file (starting time, last submission, time difference)
    -Statistics of the number of modifications made to each configuration file (number of submissions after the start time)
    -The average number of modifications for all projects, as well as the projects with the highest and lowest number of modifications

    Note: If there is a review start time for the project, it will be used as the starting time, and only submissions after the starting time will be considered for statistics,
    Calculate the difference in days between the last submission and the start time, and no longer consider the first submission as a submission record.
    """
    # if do_fetch:
        # fetch_commit_history(commit_history_file,output_file)

    # -------------------------------
    # 3. Output overall statistical information to the console

    def drop_commit_content_columns(valid_repos, input_file: str, output_file: str):
        if not os.path.exists(input_file):
            print(f"❌ File does not exist: {input_file}")
            return

        df = pd.read_csv(input_file)

        # Unified format to ensure successful matching
        df['Repository'] = df['Repository'].str.strip().str.lower()
        valid_repos = set(r.strip().lower() for r in valid_repos)

        # Only keep the warehouse in valid_repos
        df = df[df['Repository'].isin(valid_repos)]

        # Delete the specified column (if it exists)
        drop_cols = ['Initial_Commit_Content', 'Last_Commit_Content']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # Save new file
        df.to_csv(output_file, index=False)
        print(f"✅ Successfully saved the file to: {output_file}, total {len (df)} lines")


    modifications_info = {}  # key: (repo, file_path), value: Modification_Count
    with open(output_file, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                repo = row["Repository"]
                path = row["File_Path"]
                mod_count = int(row["Modification_Count"])
                # modifications_info[(repo, path)] = mod_count
                if repo in valid_repos:
                    modifications_info[(repo, path)] = mod_count
            except Exception as e:
                print(f"[WARN] parsing failed: {row}, error: {e}")
                continue

    total_files = len(modifications_info)
    total_modifications = sum(modifications_info.values())
    avg_modifications = total_modifications / total_files if total_files > 0 else 0
    max_modifications = max(modifications_info.values()) if modifications_info else 0
    min_modifications = min(modifications_info.values()) if modifications_info else 0

    projects_max = [(repo, file_path) for (repo, file_path), mod in modifications_info.items() if mod == max_modifications]
    projects_min = [(repo, file_path) for (repo, file_path), mod in modifications_info.items() if mod == min_modifications]

    print(len(valid_repos))
    print("Total number of projects：", total_files)
    print("Average number of modifications：{:.2f}".format(avg_modifications))
    print("Most modifications：", max_modifications, "，Repo：", projects_max)
    print("Least modifications：", min_modifications, "，Repo：", projects_min)

    drop_commit_content_columns(valid_repos, output_file, "data/commit_statistics.csv")


def analyze_config_change_dimention(valid_repos,change_dimention):
    df = pd.read_csv(change_dimention)

    total = 0
    fail_to_parse = 0

    value_list = value_list_for_required
    config_map = config_map_for_required

    change_dimention_names = ["action", "action_name", "trigger", "other_action_parameter"]
    change_dimention_names.extend(value_list)
    config_repo_num_list = [0 for i in range(len(change_dimention_names))]

    for index, row in df.iterrows():
        modification = row['Modification_Count']
        if modification == 0:
            continue;
        repo = row['Repository']
        if repo not in valid_repos:
            continue;
        total += 1
        initial_content = row['Initial_Commit_Content']
        lastest_content = row['Last_Commit_Content']
        review_action = get_review_action_name(lastest_content)
        differences = compare_config_content(initial_content, lastest_content, review_action)
        if review_action == "freeedcom/ai-codereviewer":
            review_action = "aidar-freeed/ai-codereviewer"

        if differences == None:
            fail_to_parse += 1
            continue;

        repo_config_change_bools = [False for i in range(len(change_dimention_names))]
        for diff in differences:
            end_point_of_config = diff[0].split(".")[-1]
            if end_point_of_config=="uses":
                repo_config_change_bools[0] = True
                continue;
            if end_point_of_config=="name":
                repo_config_change_bools[1] = True
                continue;
            if diff[0].startswith("on") \
               or end_point_of_config=="if":
                repo_config_change_bools[2] = True
                continue;

            if end_point_of_config not in config_map[review_action].keys():
                repo_config_change_bools[3] = True
                continue;
            config_type = config_map[review_action][end_point_of_config]
            if config_type!="required_item":
                config_type_index = value_list.index(config_type)
                repo_config_change_bools[4+config_type_index] = True

        for index in range(len(repo_config_change_bools)):
            if repo_config_change_bools[index]:
                config_repo_num_list[index]+=1

    print(f"total\t{total}")
    print(f"fail_to_parse\t{fail_to_parse}")
    for type,num in zip(change_dimention_names,config_repo_num_list):
        print(f"{type}\t{num}")


def write_results_to_file(file_path, results):
    if not results:
        return
    with open(file_path, mode="a", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(results)


def analyze_final_config_dimention(valid_repos,change_dimention,output_file):
    df = pd.read_csv(change_dimention)

    total = 0
    fail_to_parse = 0
    success_to_parse = 0
    make_config = 0
    config_repo_num_list = [0 for i in value_list_for_required]

    if os.path.exists(output_file):
        os.remove(output_file)
    write_results_to_file(output_file,[["Repository","Last_Commit_Content","Fail_To_Parse",
                                                "task_config_parameter",
                                                "task_trigger_parameter",
                                                "input_file_setting",
                                                "required_item",
                                                "model_env",
                                                "model",
                                                "model_parameter",
                                                "prompt",
                                                "prompt_setting",
                                                "output_setting",
                                                "other"
                                                ]])

    for index, row in df.iterrows():
        repo = row['Repository']
        if repo not in valid_repos:
            continue

        total += 1
        lastest_content = row['Last_Commit_Content']
        review_action = get_review_action_name(lastest_content)
        lastest_config_data = get_config_data(lastest_content)
        if lastest_config_data==None:
            fail_to_parse += 1
            result = [repo,lastest_content,True]
            result.extend([False for i in value_list_for_required])
            write_results_to_file(output_file, [result])
            continue;

        lastest_config = focus_config_data(lastest_config_data, review_action)
        if lastest_config==None:
            fail_to_parse += 1
            result = [repo, lastest_content, True]
            result.extend([False for i in value_list_for_required])
            write_results_to_file(output_file, [result])
            continue;

        repo_config_list = [False for i in value_list_for_required]
        success_to_parse += 1
        is_make_config = False
        for config in lastest_config:
            end_point_of_config = config.split(".")[-1]
            if review_action=="freeedcom/ai-codereviewer":
                review_action="aidar-freeed/ai-codereviewer"
            action_config_map = config_map_for_required[review_action]
            if end_point_of_config not in action_config_map:
                continue;
            config_type = config_map_for_required[review_action][end_point_of_config]
            if config_type!="required_item":
                is_make_config=True
                config_type_index = value_list_for_required.index(config_type)
                repo_config_list[config_type_index] = True

        if is_make_config:
            make_config +=1

        result = [repo, lastest_content, False]
        result.extend(repo_config_list)
        write_results_to_file(output_file, [result])
        for index in range(len(repo_config_list)):
            if repo_config_list[index]:
                config_repo_num_list[index]+=1

    print(f"total\t{total}")
    print(f"fail_to_parse\t{fail_to_parse}")
    print(f"success_to_parse\t{success_to_parse}")
    print(f"make_config\t{make_config}")
    for type,num in zip(value_list_for_required,config_repo_num_list):
        print(f"{type}\t{num}")

def gather_valid_repos() -> Set[str]:
    """
    Extract the repo names of all valid projects from multiple repo naming files.
    """
    valid_repos = set()
    for AI_reviewer in all_action:
        valid_action_repos = set()
        AI_reviewer_refined_for_path = AI_reviewer.replace("/", "_")
        repo_file = f"{BASE_DIR}/_data/{AI_reviewer_refined_for_path}/crawled_data/repo_mentioning(pr>=50).csv"
        with open(repo_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                valid_action_repos.add(row["Repository"])
        print(f"[INFO] Gathered {len(valid_action_repos)} unique valid repos from {repo_file} .")
        valid_repos.update(valid_action_repos)
    return valid_repos


if __name__ == "__main__":

    analysis_package = f"{BASE_DIR}/analyze_config_file/data"

    # File modification history:
    config_commit_history = f"{analysis_package}/config_commit_history.csv"

    # Crawl the earliest AI review registration time for a project:
    output_file = f"{analysis_package}/review_time.csv"

    # Analyze and analyze the modification history (extract the modification records of a config file)
    valid_repos = gather_valid_repos()

    config_commit_history = f"{analysis_package}/config_commit_history.csv"
    review_time_file = f"{analysis_package}/review_time.csv"
    output_file = f"{analysis_package}/code_diff_stats.csv"
    # analyze_code_diff_stats(valid_repos, config_commit_history, output_file)

    change_dimention = f"{analysis_package}/code_diff_stats.csv"
    output_file = "data/final_config_stats.csv"
    # Analyze the final configuration file version.
    analyze_final_config_dimention(valid_repos,change_dimention,output_file)
    # Analyze changes in the configuration file.
    # analyze_config_change_dimention(valid_repos,change_dimention)
