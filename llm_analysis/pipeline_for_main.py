import pandas as pd
import os
from pathlib import Path
import extract_suggestion as suggestion_ectractor
import analyze_addressed_status as address_analyzer
from config import BASE_DIR

review_action = [
    "anc95/ChatGPT-CodeReview",
    "mattzcarey/code-review-gpt",
    "coderabbitai/ai-pr-reviewer",
    "aidar-freeed/ai-codereviewer",
]

def collect_commemt(review_file: str, source_label: str) -> pd.DataFrame:
    """
    Based on the path of the review_file, read the corresponding LLM input file, generate a change label (Change_Label) for each comment, and retain a consistent Comment_ID.
    """
    if not os.path.exists(review_file):
        print(f"[WARN] review_file does not existbu: {review_file}")
        return pd.DataFrame()

    df_main = pd.read_csv(review_file)

    # ✅ Construct the Comment_ID.
    if source_label == "mattzcarey/code-review-gpt" and "File_Review_Index" in df_main.columns:
        df_main["Comment_ID"] = df_main["Comment_URL"].astype(str) + "-" + df_main["File_Review_Index"].astype(str)
    else:
        df_main["Comment_ID"] = df_main["Comment_URL"]

    llm_input_directory = review_file.replace("feature_analysis", "llm_input").rsplit("/", 1)[0]

    # ✅ Locate the llm_input file path.
    if source_label in ["anc95/ChatGPT-CodeReview","mattzcarey/code-review-gpt"]:
        llm_input_file = os.path.join(llm_input_directory,"valid_reviews(diff_reshaped)(llm_input)(consider_path).csv")
    elif source_label.lower() == "human":
        llm_input_file = os.path.join(llm_input_directory,"valid_human_reviews(llm_input)(consider_path).csv")
    else:
        llm_input_file = os.path.join(llm_input_directory,"valid_reviews(llm_input)(consider_path).csv")

    if not os.path.exists(llm_input_file):
        print(f"[WARN] Corresponding LLM input file not found: {llm_input_file}")
        return pd.DataFrame()

    df_input = pd.read_csv(llm_input_file)

    # ✅ Bind the Comment_ID
    if source_label == "mattzcarey/code-review-gpt":
        if len(df_main) != len(df_input):
            raise ValueError(f"[ERROR] Line count mismatch: review_file has {len(df_main)} lines, llm_input has {len(df_input)} lines")
        df_input["Comment_ID"] = df_main["Comment_ID"].values
    else:
        comment_map = df_main[["Comment_URL", "Comment_ID"]].drop_duplicates()
        df_input = pd.merge(df_input, comment_map, on="Comment_URL", how="left")

    # ✅ Generate the final output (retain original fields, replace Comment_URL with Comment_ID).
    cols = df_input.columns.tolist()
    cols.remove("Comment_URL")
    df_output = df_input[cols].copy()
    df_output["Source"] = source_label

    return df_output

def main_for_collect_commemt():
    all_dfs = []
    for action in review_action:
        action_ref = action.replace("/", "_")
        feature_analysis_directory = os.path.join(BASE_DIR, "_data", action_ref, "feature_analysis")
        action_review_file = os.path.join(feature_analysis_directory, "final_action_reviews.csv")
        human_review_file = os.path.join(feature_analysis_directory, "final_human_reviews.csv")

        df_action = collect_commemt(action_review_file, source_label=action)
        df_human = collect_commemt(human_review_file, source_label="Human")

        all_dfs.extend([df_action, df_human])

    merged_df = pd.concat(all_dfs, ignore_index=True)
    output_path = os.path.join("./input", "reviews(llm_input)(consider_path).csv")
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Save as：{output_path}")


def main_for_analyze_sentiment():
    ####### Subtask 1: Extract issues or suggestions from the reviews #######
    review_file = f"input/reviews(llm_input)(consider_path).csv"
    file_name = Path(review_file).stem
    gpt_version = suggestion_ectractor.llm_model
    suggestion_p_version = "3.12"
    suggestion_run_version = "1"
    output_package = f"output/{file_name}"
    if not os.path.exists(output_package):
        os.makedirs(output_package)
    output_file = f"{output_package}/Suggestion_{gpt_version}_p={suggestion_p_version}({suggestion_run_version}).csv"
    output_record_file = output_file.replace(".csv", "(record).txt")
    suggestion_ectractor.indentify_suggestion_in_comment(review_file, output_file, output_record_file)
    format_output_file = output_file.replace(".csv", "(f).csv")
    suggestion_ectractor.format_suggestion(output_file, format_output_file)

    ####### Subtask 2: Extract issues or suggestions from the reviews #######
    suggestion_items_file = format_output_file
    gpt_version = address_analyzer.llm_model
    address_p_version = "4.7"
    address_run_version = "1"
    output_file = f"{output_package}/Addressed_{gpt_version}_p={address_p_version}({address_run_version})_based_Suggestion_{gpt_version}_p={suggestion_p_version}({suggestion_run_version}).csv"
    output_record_file = output_file.replace(".csv", "(record).txt")
    address_analyzer.analyze_review_abbressed_state(review_file, suggestion_items_file,
                                                    output_file, output_record_file)

    gpt_analysis_file = output_file
    format_output_file = gpt_analysis_file.replace(".csv", "(f).csv")
    address_analyzer.format_abbressed_state(gpt_analysis_file, format_output_file)


if __name__ == "__main__":

    # main_for_collect_commemt()
    main_for_analyze_sentiment()