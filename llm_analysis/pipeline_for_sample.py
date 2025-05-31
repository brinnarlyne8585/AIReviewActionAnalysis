import time
import pandas as pd
from config import BASE_DIR
import os
from pathlib import Path
import extract_suggestion as suggestion_ectractor
import analyze_addressed_status as address_analyzer
from call_llm import get_model_function


def pipeline_for_review_file(review_file,v):

    ####### Subtask 1: Extract issues or suggestions from the reviews #######
    file_name = Path(review_file).stem
    task1_gpt = suggestion_ectractor.llm_model
    suggestion_p_version = "3.12"
    suggestion_run_version = f"{v}"
    output_package = f"output/{file_name}"
    if not os.path.exists(output_package):
        os.makedirs(output_package)
    output_file = f"{output_package}/Suggestion_{task1_gpt}_p={suggestion_p_version}({suggestion_run_version}).csv"
    output_record_file = output_file.replace(".csv", "(record).txt")
    suggestion_ectractor.indentify_suggestion_in_comment(review_file,output_file, output_record_file)
    format_output_file = output_file.replace(".csv", "(f).csv")
    suggestion_ectractor.format_suggestion(output_file, format_output_file)

    ####### Subtask 2: Extract issues or suggestions from the reviews #######
    suggestion_items_file = format_output_file
    task2_gpt = address_analyzer.llm_model
    address_p_version = "4.7"
    address_run_version = f"{v}"
    output_file = f"{output_package}/Addressed_{task2_gpt}_p={address_p_version}({address_run_version})_based_Suggestion_{task1_gpt}_p={suggestion_p_version}({suggestion_run_version}).csv"
    output_record_file = output_file.replace(".csv", "(record).txt")
    address_analyzer.analyze_review_abbressed_state(review_file, suggestion_items_file,
                                                    output_file, output_record_file)

    gpt_analysis_file = output_file
    format_output_file = gpt_analysis_file.replace(".csv", "(f).csv")
    address_analyzer.format_abbressed_state(gpt_analysis_file, format_output_file)


def run_all_batches():
    review_file_list = [
        "./input/sampled_human_review.csv",
        "./input/sampled_patch_level_review.csv",
        "./input/sampled_file_level_review.csv",
    ]

    normal_runs = [
        ("openai-gpt-4.1", "openai-o3-mini"),
        ("openai-gpt-4.1", "openai-o4-mini"),
        ("openai-gpt-4.1", "openai-gpt-4o"),

        ("deepseek-v3", "openai-o3-mini"),
        ("deepseek-v3", "openai-o4-mini"),
        ("deepseek-v3", "openai-gpt-4o"),

        ("claude-3-7-sonnet", "openai-o3-mini"),
        ("claude-3-7-sonnet", "openai-o4-mini"),
        ("claude-3-7-sonnet", "openai-gpt-4o"),
    ]

    deferred_runs = [
        ("openai-gpt-4.1", "deepseek-r1"),
        ("deepseek-v3", "deepseek-r1"),
        ("claude-3-7-sonnet", "deepseek-r1"),
    ]

    def run_batch(sugg_model, addr_model):
        suggestion_ectractor.llm_model = sugg_model
        suggestion_ectractor.response_function = get_model_function(sugg_model, input_type="prompt")
        address_analyzer.llm_model = addr_model
        address_analyzer.response_function = get_model_function(addr_model, input_type="prompt")
        for v in [1,2,3,4,5]:
            for review_file in review_file_list:
                pipeline_for_review_file(review_file, v)

    for sugg_model, addr_model in normal_runs:
        run_batch(sugg_model, addr_model)

    for sugg_model, addr_model in deferred_runs:
        run_batch(sugg_model, addr_model)

if __name__ == "__main__":

    run_all_batches()

